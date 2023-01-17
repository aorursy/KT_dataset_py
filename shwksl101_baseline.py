# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import logging
from nltk import word_tokenize
import pandas as pd
import numpy as np
import gc
import os
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from torch import optim
import torchtext
import random
text = torchtext.data.Field(lower=True, batch_first=True, tokenize=word_tokenize)
qid = torchtext.data.Field()
target = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
train = torchtext.data.TabularDataset(path='../input/train.csv', format='csv',fields={'question_text': ('text',text),'target': ('target',target)})
test = torchtext.data.TabularDataset(path='../input/test.csv', format='csv',fields={'qid': ('qid', qid),'question_text': ('text', text)})
print(os.listdir("../input/embeddings/wiki-news-300d-1M"))
text.build_vocab(train, test, min_freq=5)
qid.build_vocab(test)
text.vocab.load_vectors(torchtext.vocab.Vectors('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'))
print(text.vocab.vectors.shape)
random.seed(1215)
train, val = train.split(split_ratio=0.9, random_state=random.getstate())
class BiLSTM(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2):
        
        super(BiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        
        if static:
            self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            dropout = dropout,
                            bidirectional=True)

        self.hidden2label = nn.Linear(hidden_dim*lstm_layer*2, 1)
    
    def forward(self, sents):
        
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
        
        return y
def training(epoch, model, eval_every, loss_func, optimizer, train_iter, val_iter, early_stop=1, warmup_epoch = 2):
    
    step = 0
    max_loss = 1e5
    no_improve_epoch = 0
    no_improve_in_previous_epoch = False
    fine_tuning = False
    train_record = []
    val_record = []
    losses = []
    
    for e in range(epoch):
        
        if e >= warmup_epoch:
            if no_improve_in_previous_epoch:
                no_improve_epoch += 1
                if no_improve_epoch >= early_stop:
                    break
            else:
                no_improve_epoch = 0
            no_improve_in_previous_epoch = True

        if not fine_tuning and e >= warmup_epoch:
            
            model.embedding.weight.requires_grad = True
            fine_tuning = True
        
        train_iter.init_epoch()
        
        for train_batch in iter(train_iter):
            step += 1
            model.train()
            
            x = train_batch.text.cuda()
            y = train_batch.target.type(torch.Tensor).cuda()
            
            model.zero_grad()
            
            pred = model.forward(x).view(-1)
            
            loss = loss_function(pred, y)
            losses.append(loss.cpu().data.numpy())
            train_record.append(loss.cpu().data.numpy())
            
            loss.backward()
            optimizer.step()
            
            if step % eval_every == 0:
                
                model.eval()
                model.zero_grad()
                val_loss = []
                
                for val_batch in iter(val_iter):
                    val_x = val_batch.text.cuda()
                    val_y = val_batch.target.type(torch.Tensor).cuda()
                    val_pred = model.forward(val_x).view(-1)
                    val_loss.append(loss_function(val_pred, val_y).cpu().data.numpy())
                
                val_record.append({'step': step, 'loss': np.mean(val_loss)})
                
                print('epcoh {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} '.format(
                            e, step, np.mean(losses), val_record[-1]['loss']))
                
                if e >= warmup_epoch:
                    
                    if val_record[-1]['loss'] <= max_loss:
                        save(m=model, info={'step': step, 'epoch': e, 'train_loss': np.mean(losses),
                                            'val_loss': val_record[-1]['loss']})
                        max_loss = val_record[-1]['loss']
                        no_improve_in_previous_epoch = False
    

def save(m, info):
    
    torch.save(info, 'best_model.info')
    torch.save(m, 'best_model.m')
    
def load():
    
    m = torch.load('best_model.m')
    info = torch.load('best_model.info')
    
    return m, info
batch_size = 128

train_iter = torchtext.data.BucketIterator(dataset=train,
                                               batch_size=batch_size,
                                               sort_key=lambda x: x.text.__len__(),
                                               shuffle=True,
                                               sort=False)

val_iter = torchtext.data.BucketIterator(dataset=val,
                                             batch_size=batch_size,
                                             sort_key=lambda x: x.text.__len__(),
                                             train=False,
                                             sort=False)

model = BiLSTM(text.vocab.vectors, lstm_layer=2, padding_idx=text.vocab.stoi[text.pad_token], hidden_dim=128).cuda()

# loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_w]).cuda())

loss_function = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-3)
training(model=model, epoch=20, eval_every=500,
         loss_func=loss_function, optimizer=optimizer, train_iter=train_iter,
        val_iter=val_iter, warmup_epoch=3, early_stop=2)
model, m_info = load()
m_info
model.lstm.flatten_parameters()
model.eval()
val_pred = []
val_true = []
val_iter.init_epoch()
for val_batch in iter(val_iter):
    val_x = val_batch.text.cuda()
    val_true += val_batch.target.data.numpy().tolist()
    val_pred += torch.sigmoid(model.forward(val_x).view(-1)).cpu().data.numpy().tolist()
tmp = [0,0,0] # idx, cur, max
delta = 0
for tmp[0] in np.arange(0.1, 0.501, 0.01):
    tmp[1] = f1_score(val_true, np.array(val_pred)>tmp[0])
    if tmp[1] > tmp[2]:
        delta = tmp[0]
        tmp[2] = tmp[1]

print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
model.eval()
model.zero_grad()
test_iter = torchtext.data.BucketIterator(dataset=test,
                                    batch_size=batch_size,
                                    sort_key=lambda x: x.text.__len__(),
                                    sort=True)
test_pred = []
test_id = []

for test_batch in iter(test_iter):
    test_x = test_batch.text.cuda()
    test_pred += torch.sigmoid(model.forward(test_x).view(-1)).cpu().data.numpy().tolist()
    test_id += test_batch.qid.view(-1).data.numpy().tolist()
    
sub_df =pd.DataFrame()
sub_df['qid'] = [qid.vocab.itos[i] for i in test_id]
sub_df['prediction'] = (np.array(test_pred) >= delta).astype(int)
sub_df.to_csv("submission.csv", index=False)