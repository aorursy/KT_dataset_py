# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

import time

import random

import numpy as np

import pandas as pd

import math

import psutil

import gc

from collections import Counter

import warnings

warnings.filterwarnings('ignore')

import sys

# from sklearn.externals import joblib

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

import time

from torch.autograd import Variable

from glob import glob

from sys import getsizeof

import os

import torch.nn as nn

from torch.optim import lr_scheduler

from torch import optim

from torchvision.datasets import ImageFolder

from torchvision.utils import make_grid

import shutil

from torchvision import transforms

from torchvision import models

from torchtext import data, datasets

from nltk import ngrams

from torchtext.vocab import GloVe, Vectors

from collections import defaultdict

data_path = '/kaggle/input/spooky-author-identification/'

import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from nltk import word_tokenize

from nltk.corpus import stopwords

from torch.utils.data import DataLoader

import nltk

stop_words = stopwords.words('english')

from torch.nn import utils as nn_utils

print (torch.cuda.is_available())
def read_data(row = None):

    df_train = pd.read_csv(data_path + 'train.zip', nrows = row)

    df_test = pd.read_csv(data_path + 'test.zip', nrows = row)

    df_sub = pd.read_csv(data_path + 'sample_submission.zip', nrows = row)

    return df_train, df_test, df_sub

df_train, df_test, df_sub = read_data()

print (df_train.shape, df_test.shape, df_sub.shape)

print (df_train.head(3), '\n\n', df_test.head(3), '\n\n', df_sub.head(3))

# 处理数据

lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(df_train.author.values)

xtrain, xvalid, ytrain, yvalid = train_test_split(df_train.text.values, y, 

                                                  stratify = y, 

                                                  random_state = 2020, 

                                                  test_size = 0.2, shuffle = True)

xtest = df_test.text.values

print (xtrain.shape)

print (xvalid.shape)

print (xtest.shape)
# # load the GloVe vectors in a dictionary:

# embeddings_word_set = set()

# def sent2vec_add_word(s):

#     words = str(s).lower()

#     words = word_tokenize(words)

#     words = [w for w in words if not w in stop_words]

#     words = [w for w in words if w.isalpha()]

#     for w in words:

#         embeddings_word_set.add(str.encode(w))

#     return 0



# temp = [sent2vec_add_word(x) for x in xtrain]

# del temp

# gc.collect()

# temp = [sent2vec_add_word(x) for x in xvalid]

# del temp

# gc.collect()

# temp = [sent2vec_add_word(x) for x in xtest]

# del temp

# gc.collect()



# # %% [code]

# # load the GloVe vectors in a dictionary:

# embeddings_index = {}

# f = open(data_path + '../glove840b300dtxt/glove.840B.300d.txt', 'rb')

# f_now = open('/kaggle/working/glove840b300dtxt_now.txt', 'wb')

# index = 0

# pre_time = time.time()

# glove840b300dtxt_now = []

# for line in f: # tqdm(f):

#     index += 1

#     values = line.split()

#     word = values[0]

#     coefs = np.asarray(values[1:], dtype='float32')

#     if word in embeddings_word_set:

#         embeddings_index[word] = coefs

#         glove840b300dtxt_now.append(line)

#     if index % 500000 == 0:

#         print ('index: {:}, time:{:}'.format(index, time.time() - pre_time))

# f_now.write(b''.join(glove840b300dtxt_now))

# f_now.close()

# f.close()

# print('Found %s word vectors.' % len(embeddings_index))



# load the GloVe vectors in a dictionary:

embeddings_word_set = set()

def sent2vec_add_word(s):

    words = str(s).lower()

    words = word_tokenize(words)

    words = [w for w in words if not w in stop_words]

    words = [w for w in words if w.isalpha()]

    for w in words:

        embeddings_word_set.add(str.encode(w))

    return 0



temp = [sent2vec_add_word(x) for x in xtrain]

del temp

gc.collect()

temp = [sent2vec_add_word(x) for x in xvalid]

del temp

gc.collect()

temp = [sent2vec_add_word(x) for x in xtest]

del temp

gc.collect()



# %% [code]

# load the GloVe vectors in a dictionary:

embeddings_index = {}

f = open('/kaggle/input/glove840b300dtxt-spooky/glove840b300dtxt_now.txt', 'rb')

index = 0

pre_time = time.time()

for line in f: # tqdm(f):

    index += 1

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    if word in embeddings_word_set:

        embeddings_index[word] = coefs

    if index % 5000 == 0:

        print ('index: {:}, time:{:}'.format(index, time.time() - pre_time))

f.close()

print('Found %s word vectors.' % len(embeddings_index))
def sent2vec(s):

    words = str(s).lower()

    words = word_tokenize(words)

    words = [w for w in words if not w in stop_words]

    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:

        try:

            torch_tmp = list(embeddings_index[str.encode(w)])

            M.append(torch_tmp)

        except:

            continue

    if len(M) == 0:

        M.append([0] * 300)

    return M



xtrain_glove = [sent2vec(x) for x in xtrain]

xvalid_glove = [sent2vec(x) for x in xvalid]

xtest_glove = [sent2vec(x) for x in xtest]
word_vector_size = len(xtrain_glove[0][0])

label_size = 3

print ('word_vector_size: {:}, label_size: {:}'.format(word_vector_size, label_size))



xtrain_lengths = torch.LongTensor([len(x) for x in xtrain_glove]).cuda()

print (xtrain_lengths.max())

max_length = 200

xtrain_torch = torch.zeros((len(xtrain_glove), max_length, word_vector_size)).float().cuda()

for idx in range(len(xtrain_glove)):

    seqlen = min(int(xtrain_lengths[idx].cpu().numpy()), max_length)

    xtrain_torch[idx, :seqlen] = torch.FloatTensor(np.array(xtrain_glove[idx])[: seqlen, :])



print (type(xtrain_torch), xtrain_torch.size())

xtrain_lengths, seq_idx = xtrain_lengths.sort(0, descending = True)

xtrain_torch = xtrain_torch[seq_idx]

if isinstance(ytrain, np.ndarray):

    ytrain = torch.from_numpy(ytrain).cuda()[seq_idx]
input_size = word_vector_size

num_classes = 3

hidden_size = 256

num_layers = 1

learning_rate = 0.03

device = 'cuda'

batch_size = 8000

drop_rate = 0.1

print ("word_vector_size: {:}".format(word_vector_size))

print ("input_size: {:}".format(input_size))

print ("hidden_size: {:}".format(hidden_size))

print ("num_layers: {:}".format(num_layers))

print ("num_classes: {:}".format(num_classes))

print ("learning_rate: {:}".format(learning_rate))

print ("batch_size: {:}".format(batch_size))

print ("drop_rate: {:}".format(drop_rate))

### 定义模型

class simpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_rate):

        super(simpleLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

        self.drop_layer = nn.Dropout(p = drop_rate)



    def forward(self, x, batch_size):

        # x shape (batch, time_step, input_size)

        # out shape (batch, time_step, output_size)

        # h_n shape (n_layers, batch, hidden_size)

        # h_c shape (n_layers, batch, hidden_size)

        # 初始化hidden和memory cell参数

        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)

        c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)



        # forward propagate lstm

        encoder_outputs_packed, (h_n, h_c) = self.lstm(x, (h0, c0))

        # print (type(encoder_outputs_packed))

        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed, batch_first=True)

        # print (type(lens_unpacked), lens_unpacked.size())

        lens_unpacked_sub = lens_unpacked.sub(1)

        # out = encoder_outputs_packed

        

        # 选取最后一个时刻的输出

        h_n = torch.transpose(h_n, 0, 1)

        # print (type(h_n), h_n.size())

        out = self.fc(h_n[:, -1, :])

        out = self.drop_layer(out)

        return out



    

model = simpleLSTM(input_size, hidden_size, num_layers, num_classes, drop_rate = drop_rate).cuda()



# loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), learning_rate)



class MyDataset(data.Dataset):

    def __init__(self, images, labels, length):

        self.images = images

        self.labels = labels

        self.length = length



    def __getitem__(self, index):#返回的是tensor

        img, target, length = self.images[index], self.labels[index], self.length[index]

        return img, target, length



    def __len__(self):

        return len(self.images)



xtrain_torch = xtrain_torch.float()

print (xtrain_torch.dtype, ytrain.dtype, xtrain_lengths.dtype)

print (xtrain_torch.size())

train_loader = DataLoader(MyDataset(xtrain_torch, ytrain, xtrain_lengths), batch_size = batch_size,shuffle=False)



total_step = len(train_loader)

start_time = time.time()

epoch_size = 2000

for epoch in range(epoch_size):  # again, normally you would NOT do 300 epochs, it is toy data

    total = 0

    correct = 0

    for i, (xtrain_batch, ytrain_batch, xtrain_length_batch) in enumerate(train_loader):

        xtrain_length_batch = torch.clamp(xtrain_length_batch, 0, 100)

        # print (xtrain_batch.size(), xtrain_length_batch.size())

        embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(xtrain_batch, xtrain_length_batch, batch_first=True)

        

        # forward pass

        outputs = model(embed_input_x_packed, len(xtrain_batch))

        # print ("outputs.size(): {:}".format(outputs.size()))

        

        _, predicted = torch.max(outputs.data, 1)

        total += xtrain_batch.size(0)

        correct += (predicted == ytrain_batch).sum().item()

        

        loss = criterion(outputs, ytrain_batch)



        # backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 20 == 0 and i == 0:

            print (i, correct, total)

            print (Counter(predicted.cpu().numpy()), Counter(ytrain_batch.cpu().numpy()),)

    if epoch % 20 == 0:

        print('Epoch [{}/{}], Accuracy: {} / {} = {:.0f}%,  Loss: {:.4f}, time: {:}'. \

              format(epoch+1, epoch_size, correct, total, (100 * correct / total), loss.item(), time.time() - start_time))
xvalid_lengths = torch.FloatTensor([len(x) for x in xvalid_glove]).cuda()

xvalid_torch = torch.zeros((len(xvalid_glove), max_length, word_vector_size)).float().cuda()

for idx in range(len(xvalid_glove)):

    seqlen = min(int(xvalid_lengths[idx].cpu().numpy()), max_length)

    xvalid_torch[idx, :seqlen] = torch.FloatTensor(np.array(xvalid_glove[idx])[: seqlen, :])



print (type(xvalid_torch), xvalid_torch.size())

xvalid_lengths, seq_idx_valid = xvalid_lengths.sort(0, descending = True)

xvalid_torch = xvalid_torch[seq_idx_valid]

if isinstance(yvalid, np.ndarray):

    yvalid = torch.from_numpy(yvalid).cuda()[seq_idx_valid]



# See what the scores are after training

with torch.no_grad():

    xvalid_length_batch = torch.clamp(xvalid_lengths, 0, 100).cuda()

#     print (type(xvalid_length_batch), xvalid_length_batch.device, xvalid_length_batch.size())

#     print (type(xvalid_torch), xvalid_torch.device, xvalid_torch.size())

    embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(xvalid_torch, xvalid_length_batch, batch_first=True)

    outputs = model(embed_input_x_packed, xvalid_torch.size(0))

    _, predicted = torch.max(outputs.data, 1)

    

    total = xvalid_torch.size(0)

    correct = (predicted == yvalid).sum().item()

print('Test Accuracy of the model on the test: {}%'.format(100 * correct / total))

print (predicted.cpu().numpy()[: 10])

print (Counter(predicted.cpu().numpy()))

print (Counter(yvalid.cpu().numpy()))

print (Counter(ytrain.cpu().numpy()))
xtest_lengths = torch.FloatTensor([len(x) for x in xtest_glove]).cuda()

xtest_torch = torch.zeros((len(xtest_glove), max_length, word_vector_size)).float().cuda()

for idx in range(len(xtest_glove)):

    seqlen = min(int(xtest_lengths[idx].cpu().numpy()), max_length)

    xtest_torch[idx, :seqlen] = torch.FloatTensor(np.array(xtest_glove[idx])[: seqlen, :])



print (type(xvalid_torch), xtest_torch.size())

xtest_lengths, seq_idx_test = xtest_lengths.sort(0, descending = True)

xtest_torch = xtest_torch[seq_idx_test]



# See what the scores are after training

with torch.no_grad():

    xtest_length_batch = torch.clamp(xtest_lengths, 0, 100).cpu()

    embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(xtest_torch, xtest_length_batch, batch_first=True)

    outputs = model(embed_input_x_packed, xtest_torch.size(0))

print (outputs.size())



dim1_softmax = nn.Softmax(dim = 1)

outputs = dim1_softmax(outputs)



test_id = df_sub['id'].values

test_id = test_id[list(seq_idx_test.cpu().numpy())]

df_sub_test = pd.DataFrame(data = outputs.cpu().numpy(), columns = ['EAP', 'HPL', 'MWS'])

df_sub_test['id'] = test_id

df_sub_test = df_sub_test[['id', 'EAP', 'HPL', 'MWS']]

df_sub_test.to_csv('/kaggle/working/sub_20200712_01.csv', index = False)

print (df_sub_test.head(3))