# Libraries installed are defined by the docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import torch
import torchtext
import random
import pdb
from time import time
from torch import nn
from sklearn.metrics import f1_score
from nltk import word_tokenize
from torch import optim

import os
print(os.listdir("../input"))

random.seed(420)
random_state = random.getstate()

# Any results you write to the current directory are saved as output.
text = torchtext.data.Field(lower=True, batch_first=True, tokenize=word_tokenize)
target = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
data = torchtext.data.TabularDataset(path='../input/train.csv', format='csv',
                                      fields={'question_text': ('text',text),
                                              'target': ('target',target)})
text.build_vocab(data, min_freq=1) # Tried 2. 1 is better
text.vocab.load_vectors(torchtext.vocab.Vectors('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'))
print(text.vocab.vectors.shape)
train, validation = data.split(split_ratio=0.9, random_state=random_state)
batch_size = 64

train_iter = torchtext.data.BucketIterator(dataset=train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: x.text.__len__(),
                                           shuffle=True,
                                           sort=False)

valid_iter = torchtext.data.BucketIterator(dataset=validation,
                                           batch_size=batch_size,
                                           sort_key=lambda x: x.text.__len__(),
                                           train=False,
                                           sort=False)
# Helper functions
def with_time(func):
    pre = time()
    res = func()
    print("Elapsed time:", time() - pre)
    return res

def x_y_from_batch(batch):
    x = batch.text.cuda()
    y = batch.target.type(torch.Tensor).cuda()
    
    return x, y
def training(
    model, epoch, loss_func, optimizer, 
    no_improve_max=1, train_iter=train_iter, valid_iter=valid_iter
):
    step = 0
    no_improve_streak = 0
    val_record = []

    for e in range(epoch):
        train_record = []
        train_iter.init_epoch()
        
        for train_batch in iter(train_iter):
            step += 1
            
            model.train().zero_grad()
            
            x, y = x_y_from_batch(train_batch)
            
            pred = model.forward(x).view(-1)
            
            loss = loss_function(pred, y)
            loss_data = loss.cpu().data.numpy()
            train_record.append(loss_data)
            loss.backward()
            optimizer.step()
            
            if step % 1000 == 0:
                print("Step {}, t loss {:.4f}".format(step, loss_data))
            
        # end of epoch
        model.eval().zero_grad()
                
        val_loss = []

        for val_batch in iter(valid_iter):
            val_x, val_y = x_y_from_batch(val_batch)

            val_pred = model.forward(val_x).view(-1)
            val_loss.append(loss_function(val_pred, val_y).cpu().data.numpy())

        val_record.append({'step': step, 'loss': np.mean(val_loss)})

        print('Epoch {} - step {} - train loss {:.4f} - valid loss {:.4f}'
              .format(e, step, np.mean(train_record), val_record[-1]['loss'])
         )
        
        # check for not improving
        if e > 0 and (val_record[-1]['loss'] >= val_record[-2]['loss']):
            no_improve_streak += 1
            
            if (no_improve_streak >= no_improve_max):
                print('Reached no improve max!')
                break
        else:
            no_improve_streak = 0
class Model(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, lstm_size=128, lstm_layers=1, 
                 lstm_dropout=0, dropout1=0, 
                 with_hidden=False, dropout2=0, static=True):
        super(Model, self).__init__()
        
        self.with_hidden = with_hidden
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        
        if static:
            self.embedding.weight.requires_grad = False
            
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=lstm_size,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout)
        
        self.dropout1 = nn.Dropout(p=dropout1)

        hidden_size = lstm_size * lstm_layers
            
        if self.with_hidden:
            hidden_size = lstm_size // 2
            
            self.hidden = nn.Linear(lstm_size * lstm_layers, hidden_size)
            self.hidden_activ = nn.Tanh()
            
            self.dropout2 = nn.Dropout(p=dropout2)
            
        self.final = nn.Linear(hidden_size, 1)
        
    def apply_lstm(self, x):
        _, (h_n, c_n) = self.lstm(x)
        # gather all the hidden states
        lstm_ = torch.cat([c_n[i, :, :] for i in range(c_n.shape[0])], dim=1)
        
        return self.dropout1(lstm_)
        
    def apply_hidden(self, x):
        hidden = self.hidden(x)
        hidden_act = self.hidden_activ(hidden)
        
        return self.dropout2(hidden_act)
    
    def forward(self, sents):
        x = torch.transpose(self.embedding(sents), dim0=1, dim1=0)

        hidden_res = self.apply_lstm(x)
        
        if self.with_hidden:
            hidden_res = self.apply_hidden(hidden_res)
        
        output = self.final(hidden_res)
        #output = self.final_activ(output)

        return output
loss_function = nn.BCEWithLogitsLoss()

def get_optimizer(model, lr=0.001):
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
def model_score(model):
    print('Preparing for evaluation...')

    model.eval()
    valid_iter.init_epoch()
    
    val_pred = []
    val_true = []

    for val_batch in iter(valid_iter):
        val_true += val_batch.target.data.numpy().tolist()
        val_pred += torch.sigmoid(
                        model.forward(val_batch.text.cuda()).view(-1)
                    ).cpu().data.numpy().tolist()

    print('Evaluation started...')
    
    max_f1 = 0
    thresh = 0

    # Computing the best threshold based on f1 score
    for thr in np.arange(0.1, 0.801, 0.01):
        curr = f1_score(val_true, np.array(val_pred) > thr)

        if curr > max_f1:
            thresh = thr
            max_f1 = curr

    print('Best threshold is {:.3f} with F1 score: {:.4f}'.format(thresh, max_f1))
# This is the best model so far. Other models results will be shown during the presentation
model = Model(
    pretrained_lm=text.vocab.vectors,
    padding_idx=text.vocab.stoi[text.pad_token],
    lstm_size=128,
    lstm_layers=1,
    dropout1=0.5,
    with_hidden=True,
    dropout2=0.5 # hidden layer dropout
).cuda()

with_time(lambda:
    training(
         model=model,
         epoch=10, # should be big, it will automatically stop when starting to overfit
         loss_func=loss_function,
         optimizer=get_optimizer(model, 0.001)
    )
)
model_score(model)
# Should be close to:
# Best threshold is 0.450 with F1 score: 0.6844
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD



