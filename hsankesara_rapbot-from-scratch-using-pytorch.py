# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import nltk
from nltk import tokenize
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
lyrics_path = '../input/eminem-dataset-show/eminem.csv'
GLOVE_DIR = '../input/glove6b/glove.6B.100d.txt'

# Any results you write to the current directory are saved as output.
use_gpu = torch.cuda.is_available()
lyrics = pd.read_csv(lyrics_path)
lyrics.head()
#lyrics = lyrics[:300]
import re
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\n", " \n ", string)
    string = re.sub(r"\\", "", string) 
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)
    if string.rfind('</p>') != -1:
        string = string[: string.rfind('</p>') + len('</p>')]
    return string.strip().lower()
rap_lyrics = lyrics['text'].apply(clean_str)
rap_lyrics = rap_lyrics.apply(lambda s : tokenize.word_tokenize(s, preserve_line=True)).tolist()
word_index = {}
rev_dict = {}
word_count = -1
for rap in rap_lyrics:
    for word in rap:
        if word not in word_index:
            word_count += 1
            word_index[word] = word_count
            rev_dict[word_count] = word
print(len(word_index))
embed_size=100
num_stacks = 3
num_inputs = 15
batch_size = 2048
training_iters = 500
display_step = 100
learning_rate = 1e-3
embeddings_index = {}
f = open(GLOVE_DIR)
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        print(word)
        pass
f.close()
print('Total %s word vectors.' % len(embeddings_index))
embedding_matrix = np.random.randn(len(word_index) + 1, embed_size)
absent_words = 0
absent_index = {}
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be randomly initialized.
        embedding_matrix[i] = embedding_vector
    else:
        absent_words += 1
print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)), '% of total words')
vocab_size = len(word_index)
## Convert words too their corresponding index
total_raps = len(rap_lyrics)
for i in range(total_raps):
    rap_len = len(rap_lyrics[i])
    for j in range(rap_len):
        rap_lyrics[i][j] = word_index[rap_lyrics[i][j]]
## Create training data
train_x = []
train_y = []
for rap  in rap_lyrics:
        m = len(rap)
        window = m - num_inputs
        for i in range(window):
            train_x.append(rap[i: i + num_inputs])
            train_y.append(rap[i + num_inputs])
train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x.shape, train_y.shape)
def get_accuracy(outputs, labels):
    total = float(labels.size()[0])
    _, predicted = torch.max(outputs, 1)
    correct_pred = (predicted == labels).float()
    accuracy = correct_pred.sum() / total * 100
    return accuracy
# Testing
get_accuracy(torch.Tensor([[1, 0, 0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]]), torch.LongTensor([1,1,1,1,2]))
def train_step(inputs, labels, optimizer, criterion):
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = emnet(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    acc_batch = get_accuracy(outputs, labels)
    return loss, acc_batch
class EmNet(nn.Module):
    def __init__(self, num_stacks, num_inputs, weights_matrix):
        super(EmNet, self).__init__()
        self.vocab_size, self.embed_size = weights_matrix.shape
        self.create_emb_layer(weights_matrix)
        self.num_stacks = num_stacks
        self.num_inputs = num_inputs
        self.Wfh = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.Wfx = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.Wih = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.Wix = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.Wch = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.Wcx = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.Woh = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.Wox = nn.ParameterList([nn.Parameter(torch.randn((embed_size, embed_size)), requires_grad =True) for i in range(num_stacks)])
        self.bf = nn.ParameterList([nn.Parameter(torch.zeros(1, embed_size), requires_grad =True) for i in range(num_stacks)])
        self.bi = nn.ParameterList([nn.Parameter(torch.zeros(1, embed_size), requires_grad =True) for i in range(num_stacks)])
        self.bc = nn.ParameterList([nn.Parameter(torch.zeros(1, embed_size), requires_grad =True) for i in range(num_stacks)])
        self.bo = nn.ParameterList([nn.Parameter(torch.zeros(1, embed_size), requires_grad =True) for i in range(num_stacks)])
        self.Wfinal = nn.Parameter(torch.randn(embed_size,  vocab_size), requires_grad =True)
        self.bfinal = nn.Parameter(torch.zeros(1, vocab_size), requires_grad =True)
        self.cin = Variable(torch.zeros(1, embed_size).cuda())
        self.hin = Variable(torch.zeros(1, embed_size).cuda())
        return
        
    def create_emb_layer(self, weights_matrix, non_trainable=False):
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix).cuda())
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return
        
    def lstm_cell(self, layer_num, prev_c, prev_h, curr_x):
        ft = F.sigmoid(prev_h.mm(self.Wfh[layer_num]) + curr_x.mm(self.Wfx[layer_num]) + self.bf[layer_num])
        it = F.sigmoid(prev_h.mm(self.Wih[layer_num]) + curr_x.mm(self.Wix[layer_num]) + self.bi[layer_num])
        Ct = F.tanh(prev_h.mm(self.Wch[layer_num]) + curr_x.mm(self.Wcx[layer_num]) + self.bc[layer_num])
        ot = F.sigmoid(prev_h.mm(self.Woh[layer_num]) + curr_x.mm(self.Wox[layer_num]) + self.bo[layer_num])
        ct = ft * prev_c + it * Ct
        ht = ot * F.tanh(ct)
        return ct, ht
    
    def lstm_layer(self, xin, cin, hin, layer_num):
        num_inp = self.num_inputs
        h = hin
        c  = cin
        next_inp = None
        for i in range(num_inp):
            curr = xin[:, i, :]
            c, h = self.lstm_cell(layer_num, c, h, curr)
            hfit = h.unsqueeze(1)
            if next_inp is None:
                next_inp = hfit
            else:
                next_inp = torch.cat((next_inp, hfit), 1)
        return next_inp
    
    def forward(self, x):
        x = self.embedding(x)
        for layer_num in range(self.num_stacks):
            x = self.lstm_layer(x, self.cin, self.hin, layer_num)
        x = x[:, -1, :]
        x = x.mm(self.Wfinal) + self.bfinal
        return x
emnet = EmNet(num_stacks, num_inputs, embedding_matrix)
if use_gpu:
    emnet = emnet.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(emnet.parameters(), learning_rate)
m = train_x.shape[0]
for epoch in range(training_iters):
    cst_total = 0
    acc_total = 0
    total_batches = int(np.ceil(m / batch_size))
    for i in range(total_batches):
        df_x = Variable(torch.LongTensor(train_x[i * batch_size : (i + 1) * batch_size]).cuda())
        df_y = Variable(torch.LongTensor(train_y[i * batch_size : (i + 1) * batch_size]).cuda())
        batch_loss, acc_batch = train_step(df_x, df_y, optimizer, criterion)
        cst_total += batch_loss
        acc_total += acc_batch
    if (epoch + 1) % display_step == 0:
        print('After ', (epoch + 1), 'iterations: Cost = %.2f' % (cst_total.data.cpu().numpy()[0] / total_batches), 'and Accuracy = ', acc_total.data.cpu().numpy()[0] / total_batches, '%' )
print('Optimiation finished!!!')
print("Let's test")
text_inp = "I reckon you ain't familiar with these here parts You know, there's a story behind"
tokens = tokenize.word_tokenize(clean_str(text_inp))[:num_inputs]
inp_vecs = Variable(torch.LongTensor([word_index[i] for i in tokens])).cuda().view(1,-1)
len_rap = 128
rap = inp_vecs
while rap.size()[1] < len_rap:
    out_vec = emnet(inp_vecs)
    out_vec = F.softmax(out_vec, 1)
    _, next_word = torch.max(out_vec, 1)
    rap = torch.cat((rap, next_word.unsqueeze(1)), 1)
    inp_vecs = torch.cat((inp_vecs[:, 1:], next_word.unsqueeze(1)), 1)
rap_string = ''
for i in range(len_rap):
    rap_string += rev_dict[rap[0, i].data.cpu().numpy()[0]] + ' '
print(rap_string)



