import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import os

import time

import gc

import random

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from keras.preprocessing import text, sequence

import torch

from torch import nn

from torch.utils import data

from torch.nn import functional as F
train = pd.read_csv('/kaggle/input/liarplus/liar-plus-master/LIAR-PLUS-master/dataset/train2.tsv',delimiter='\t',header=None,names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",

                                            "barely-true", "false", "half-true", "mostly-true", "pants_on_fire", "context/venue",'justification'])

val = pd.read_csv('/kaggle/input/liarplus/liar-plus-master/LIAR-PLUS-master/dataset/val2.tsv',delimiter='\t',header=None,names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",

                                            "barely-true", "false", "half-true", "mostly-true", "pants_on_fire", "context/venue",'justification'])

test = pd.read_csv('/kaggle/input/liarplus/liar-plus-master/LIAR-PLUS-master/dataset/test2.tsv',delimiter='\t',header=None,names = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",

                                            "barely-true", "false", "half-true", "mostly-true", "pants_on_fire", "context/venue",'justification'])
GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 220
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words
def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def train_model(model, train, val ,test, loss_fn, output_dim, lr=0.001, batch_size=512, n_epochs=4):

    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]

    optimizer = torch.optim.Adam(param_lrs, lr=lr)



    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

    

    all_test_preds = []

    

    for epoch in range(n_epochs):

        start_time = time.time()

        

        scheduler.step()

        

        model.train()

        avg_loss = 0.

        

        

        for data in train_loader:

            x_batch = data[:-1]

            y_batch = data[-1]



            y_pred = model(*x_batch)  

            y_pred=y_pred.squeeze()

            loss = loss_fn(y_pred, y_batch)



            optimizer.zero_grad()

            loss.backward()



            optimizer.step()

            

        

        

        model.eval()

        val_preds = np.zeros((len(val), output_dim))

        for data in val_loader:

            x_batch = data[:-1]

            y_batch = data[-1]



            y_pred = model(*x_batch)  

            y_pred=y_pred.squeeze()

            loss = loss_fn(y_pred, y_batch)          

            avg_loss += loss.item() / len(train_loader)

        

        elapsed_time = time.time() - start_time

        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, elapsed_time))

        

    

    

    model.eval()

    test_preds = np.zeros((len(test), output_dim))

     

    for i, x_batch in enumerate(test_loader):

        y_pred = (model(*x_batch).detach().cpu().numpy())

        test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred



    all_test_preds.append(test_preds)

        

    return test_preds
class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix):

        super(NeuralNet, self).__init__()

        embed_size = embedding_matrix.shape[1]

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False



        

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

    

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 6)

        



        

    def forward(self, x):

        h_embedding = self.embedding(x)

        

        h_lstm1, _ = self.lstm1(h_embedding)

        h_lstm2, _ = self.lstm2(h_lstm1)

        

        avg_pool = torch.mean(h_lstm2, 1)



        max_pool, _ = torch.max(h_lstm2, 1)

        

        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = F.relu(self.linear1(h_conc))

        h_conc_linear2  = F.relu(self.linear2(h_conc))

        

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        

        result = self.linear_out(hidden)

        result = F.log_softmax(result, dim=1)

        

        return result
def preprocess(data):



    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
pd.unique(train.label)
multi_dict = {'false':0 , 'pants-fire':1 , 'barely-true':2, 'half-true':3 , 'mostly-true':4 ,'true':5}
train['multi_label'] = train.label.apply(lambda x: multi_dict[x] )

val['multi_label'] = val.label.apply(lambda x: multi_dict[x] )

test['multi_label'] = test.label.apply(lambda x: multi_dict[x] )
x_train = preprocess(train['statement'])

y_train = train['multi_label']



x_val = preprocess(val['statement'])

y_val = val['multi_label']





#y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

x_test = preprocess(test['statement'])
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) +list(x_val)+ list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_val = tokenizer.texts_to_sequences(x_val)



x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

x_val = sequence.pad_sequences(x_val,maxlen=MAX_LEN)
max_features=None

max_features = max_features or len(tokenizer.word_index) + 1

max_features
glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))
embedding_matrix = glove_matrix
x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()

x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()

y_train_torch = torch.tensor(y_train, dtype=torch.long).cuda()

x_val_torch = torch.tensor(x_val , dtype=torch.long).cuda()

y_val_torch = torch.tensor(y_val, dtype=torch.long).cuda()
train_dataset = data.TensorDataset(x_train_torch, y_train_torch)

test_dataset = data.TensorDataset(x_test_torch)

val_dataset = data.TensorDataset(x_val_torch,y_val_torch)



all_test_preds = []



model = NeuralNet(embedding_matrix)

model.cuda()



test_preds = train_model(model, train_dataset, val_dataset ,test_dataset, output_dim=6,

                         loss_fn=nn.NLLLoss(),n_epochs=10)

all_test_preds.append(test_preds)
top_p,top_cat = torch.exp(torch.tensor(all_test_preds[0])).topk(1,dim=1)

from sklearn.metrics import accuracy_score

accuracy_score(test['multi_label'],top_cat)