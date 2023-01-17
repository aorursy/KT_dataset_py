import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import nltk
import random
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
data = pd.read_csv("../input/annotated-gmb-corpus/GMB_dataset.txt", sep="\t", header=None, encoding="latin1")
#The dataset does not have any header currently. We can use the first row as a header as it has the relevant headings.
#We will make the first row as the heading, remove the first row and re-index the dataset
data.columns = data.iloc[0]
data = data[1:]
data.columns = ['Index','Sentence #','Word','POS','Tag']
data = data.reset_index(drop=True)
data.head()
#We have 66161 samples and 5 features. We will understand them in detail in the exploration step.
#Lets check for any missing values in the dataset
data.info()
# A class to retrieve the sentences from the dataset
class getsentence(object):
    
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
getter = getsentence(data)
sentences = getter.sentences
#This is how a sentence will look like. 
print(sentences[0])
data_sent, data_pos, data_tag = [], [], []
for s in sentences:
    a, b, c = list(zip(*s))
    data_sent.append(a)
    data_pos.append(b)
    data_tag.append(c)
flatten = lambda l: [item for sublist in l for item in sublist]
vocab = list(set(flatten(data_sent)))
posset = list(set(flatten(data_pos)))
tagset = list(set(flatten(data_tag)))
word2index={'<UNK>' : 0, '<DUMMY>' : 1} # dummy token is for start or end of sentence
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
index2word = {v:k for k, v in word2index.items()}

pos2index = {'<UNK>' : 0, '<DUMMY>' : 1}
for pos in posset:
    if pos2index.get(pos) is None:
        pos2index[pos] = len(pos2index)
index2pos={v:k for k, v in pos2index.items()}

tag2index = {'<UNK>' : 0, '<DUMMY>' : 1}
for tag in tagset:
    if tag2index.get(tag) is None:
        tag2index[tag] = len(tag2index)
index2tag={v:k for k, v in tag2index.items()}
WINDOW_SIZE = 2
windows = []
for sample in list(zip(data_sent, data_pos, data_tag)):#pos
    dummy = ['<DUMMY>'] * WINDOW_SIZE
    window = list(nltk.ngrams(dummy + list(sample[0]) + dummy, WINDOW_SIZE * 2 + 1))
    window_pos = list(nltk.ngrams(dummy + list(sample[1]) + dummy, WINDOW_SIZE * 2 + 1))
    windows.extend([[list(window[i]), list(window_pos[i]), sample[2][i]] for i in range(len(sample[0]))])
random.shuffle(windows)

train_data = windows[:int(len(windows) * 0.8)]
test_data = windows[int(len(windows) * 0.8):]
def getBatch(batch_size, train_data):
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
def prepare_word(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(torch.LongTensor(idxs))

def prepare_pos(poss, pos2index):
    idxs = list(map(lambda p: pos2index[p] if pos2index.get(p) is not None else pos2index["<UNK>"], poss))
    return Variable(torch.LongTensor(idxs))

def prepare_tag(tag,tag2index):
    return Variable(torch.LongTensor([tag2index[tag]]))
BATCH_SIZE = 128
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 200
EPOCH = 100
LEARNING_RATE = 0.001
class WindowClassifier(nn.Module): 
    def __init__(self, vocab_size, pos_size, embedding_size, window_size, hidden_size, output_size, batch_size):

        super(WindowClassifier, self).__init__()
        
        self.batch_size = batch_size
        
        self.word_embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_embed = nn.Embedding(pos_size, embedding_size)
        
        self.h_layer1 = nn.Linear(2 * embedding_size * (window_size * 2 + 1), hidden_size)
        self.h_layer2 = nn.Linear(hidden_size, hidden_size)
        self.o_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        
        self.loss_function = nn.CrossEntropyLoss(reduce=False)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        
    def forward(self, inputs, pos, is_training=False): 
        embeds = self.word_embed(inputs) # BxWxD
        concated = embeds.view(-1, embeds.size(1) * embeds.size(2)) # Bx(W*D)
        
        pos_embeds = self.pos_embed(pos)
        pos_concated = embeds.view(-1, pos_embeds.size(1) * pos_embeds.size(2))

        x = torch.cat([concated, pos_concated], dim=1)

        h0 = F.sigmoid(self.h_layer1(x))
        if is_training:
            h0 = self.dropout(h0)
        h1 = F.sigmoid(self.h_layer2(h0))
        if is_training:
            h1 = self.dropout(h1)
        out = self.softmax(self.o_layer(h1))
        return out
    
    def fit(self, X, P, Y, W=None):
        if W is None:
            W = np.ones((len(X),))
        data = list(zip(X, P, Y))
        for epoch in range(EPOCH):
            losses = []
            for i, batch in enumerate(getBatch(self.batch_size, data)):
                x, p, y = list(zip(*batch))
                w = W[i * self.batch_size : min(len(X), (i + 1) * self.batch_size)]
                inputs = torch.cat([prepare_word(sent, word2index).view(1, -1) for sent in x])
                pos = torch.cat([prepare_pos(sent_pos, pos2index).view(1, -1) for sent_pos in p])
                targets = torch.cat([prepare_tag(tag, tag2index) for tag in y])
            #for (x,y) in data:
            #    inputs = torch.cat([prepare_sequence(x, word2index).view(1, -1)])
            #    targets = torch.cat([prepare_tag(y, tag2index)])
                self.zero_grad()
                preds = self.forward(inputs, pos, is_training=True)
                loss = (torch.from_numpy(w).t() * self.loss_function(preds, targets)).sum()
                losses.append(loss.data.tolist())
                loss.backward()
                self.optimizer.step()
            print("[%d/%d] mean_loss : %0.9f" %(epoch, EPOCH, np.mean(losses)))
            losses = []
            
    def predict(self, X, P):
        out = []
        data = list(zip(X, P))
        for i, batch in enumerate(getBatch(self.batch_size, data)):
            x, p =list(zip(*batch))
            inputs = torch.cat([prepare_word(sent, word2index).view(1, -1) for sent in x])
            pos = torch.cat([prepare_pos(sent_pos, pos2index).view(1, -1) for sent_pos in p])
            out += self.forward(inputs, pos).max(1)[1].data.tolist()
        return out
model = WindowClassifier(len(word2index), len(pos2index), EMBEDDING_SIZE, WINDOW_SIZE, HIDDEN_SIZE, len(tag2index), BATCH_SIZE)
X_train, p_train, y_train = list(zip(*train_data))
X_test, p_test, y_test = list(zip(*test_data))

class_weight = compute_class_weight('balanced', tagset, y_train)

W = np.array([class_weight[tag2index[t] - 2] for t in y_train])
#W = np.ones((len(X_train),))

model.fit(X_train, p_train, y_train, W)
preds = []
trues = []

for test in test_data:
    x, p, y = test[0], test[1], test[2]
    input_ = prepare_word(x, word2index).view(1, -1)
    pos_ = prepare_pos(p, pos2index).view(1, -1)
    i = model(input_, pos_).max(1)[1]
    pred = index2tag[i.data.tolist()[0]]
    
    preds.append(pred)
    trues.append(y)

print(accuracy_score(preds, trues))
print(classification_report(preds, trues))
vocab = list(set(flatten(data_sent)))
posset = list(set(flatten(data_pos)))
tagset = list(set(flatten(data_tag)))

charset = list(set(flatten([''.join(sent) for sent in data_sent])))
word2index = {'<UNK>' : 0, '<DUMMY>' : 1} # dummy token is for start or end of sentence
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
index2word = {v:k for k, v in word2index.items()}

pos2index = {'<UNK>' : 0, '<DUMMY>' : 1}
for pos in posset:
    if pos2index.get(pos) is None:
        pos2index[pos] = len(pos2index)
index2pos = {v:k for k, v in pos2index.items()}

tag2index = {}
for tag in tagset:
    if tag2index.get(tag) is None:
        tag2index[tag] = len(tag2index)
index2tag = {v:k for k, v in tag2index.items()}

char2index = {'<UNK>' : 0, '<DUMMY>' : 1}
for char in charset:
    if char2index.get(char) is None:
        char2index[char] = len(char2index)
index2char = {v:k for k, v in char2index.items()}
WINDOW_SIZE = 2
windows = []
for sample in list(zip(data_sent, data_pos, data_tag)):#pos
    dummy = ['<DUMMY>'] * WINDOW_SIZE
    window = list(nltk.ngrams(dummy + list(sample[0]) + dummy, WINDOW_SIZE * 2 + 1))
    window_pos = list(nltk.ngrams(dummy + list(sample[1]) + dummy, WINDOW_SIZE * 2 + 1))
    windows.extend([[list(window[i]), list(sample[0][i]), list(window_pos[i]), sample[2][i]] for i in range(len(sample[0]))])
random.shuffle(windows)

train_data = windows[:int(len(windows) * 0.9)]
test_data = windows[int(len(windows) * 0.9):]
def getBatch(batch_size, train_data):
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
def prepare_word(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(torch.LongTensor(idxs))

def prepare_pos(poss, pos2index):
    idxs = list(map(lambda p: pos2index[p] if pos2index.get(p) is not None else pos2index["<UNK>"], poss))
    return Variable(torch.LongTensor(idxs))

def prepare_tag(tag, tag2index):
    return Variable(torch.LongTensor([tag2index[tag]]))

def prepare_char(char, char2index):
    idxs = list(map(lambda c: char2index[c] if char2index.get(c) is not None else char2index["<UNK>"], char))
    if len(idxs) < 15:
        idxs += [char2index["<DUMMY>"]] * (15 - len(idxs))
    return Variable(torch.LongTensor(idxs[:15]))
BATCH_SIZE = 128
WORD_EMBEDDING_SIZE = 100
CHAR_EMBEDDING_SIZE =100
POS_EMBEDDING_SIZE = 100

HIDDEN_SIZE = 200
EPOCH = 100
LEARNING_RATE = 0.002
class WindowClassifierWithCharEmb(nn.Module): 
    def __init__(self, vocab_size, char_size, pos_size, word_embedding_size, char_embedding_size, pos_embedding_size, window_size, hidden_size, output_size, batch_size):

        super(WindowClassifierWithCharEmb, self).__init__()
        
        self.batch_size = batch_size
        
        self.word_embed = nn.Embedding(vocab_size, word_embedding_size)
        self.char_embed = nn.Embedding(char_size, char_embedding_size)
        self.pos_embed = nn.Embedding(pos_size, pos_embedding_size)
        
        self.h_layer1 = nn.Linear(2 * hidden_size + word_embedding_size * (window_size * 2 + 1) + pos_embedding_size * (window_size * 2 + 1), hidden_size)
        self.h_layer2 = nn.Linear(hidden_size, hidden_size)
        self.o_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(char_embedding_size, hidden_size, bidirectional=True)#word_len
        
        self.loss_function = nn.CrossEntropyLoss(reduce=False)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        
    def forward(self, inputs, char, pos, is_training=False): 
        embeds = self.word_embed(inputs) # BxWxD
        concated = embeds.view(-1, embeds.size(1) * embeds.size(2)) # Bx(W*D)
        
        char_embeds = self.char_embed(char)
        _1, _2 = self.lstm(char_embeds)
        char_embeds = _1[:,-1,:]
        
        pos_embeds = self.pos_embed(pos)
        pos_concated = pos_embeds.view(-1, pos_embeds.size(1) * pos_embeds.size(2))
        x = torch.cat([concated, char_embeds, pos_concated], dim=1)

        h0 = F.sigmoid(self.h_layer1(x))
        if is_training:
            h0 = self.dropout(h0)
        h1 = F.sigmoid(self.h_layer2(h0))
        if is_training:
            h1 = self.dropout(h1)
        out = self.softmax(self.o_layer(h1))
        return out
    
    def fit(self, X, C, P, Y, W=None):
        if W is None:
            W = np.ones((len(X),))
        data = list(zip(X, C, P, Y))
        for epoch in range(EPOCH):
            losses = []
            for i, batch in enumerate(getBatch(self.batch_size, data)):
                x, c, p, y = list(zip(*batch))
                w = W[i * self.batch_size : min(len(X), (i + 1) * self.batch_size)]
                inputs = torch.cat([prepare_word(sent, word2index).view(1, -1) for sent in x])
                char = torch.cat([prepare_char(sent_char, char2index).view(1, -1) for sent_char in c])
                pos = torch.cat([prepare_pos(sent_pos, pos2index).view(1, -1) for sent_pos in p])
                targets = torch.cat([prepare_tag(tag, tag2index) for tag in y])
            #for (x,y) in data:
            #    inputs = torch.cat([prepare_sequence(x, word2index).view(1, -1)])
            #    targets = torch.cat([prepare_tag(y, tag2index)])
                self.zero_grad()
                preds = self.forward(inputs, char, pos, is_training=True)
                loss = (torch.from_numpy(w).t() * self.loss_function(preds, targets)).sum()
                losses.append(loss.data.tolist())
                loss.backward()
                self.optimizer.step()
            print("[%d/%d] mean_loss : %0.9f" %(epoch, EPOCH, np.mean(losses)))
            losses = []
            
    def predict(self, X, C, P):
        out = []
        data = list(zip(X, C, P))
        for i, batch in enumerate(getBatch(self.batch_size, data)):
            x, c, p =list(zip(*batch))
            inputs = torch.cat([prepare_word(sent, word2index).view(1, -1) for sent in x])
            char = torch.cat([prepare_char(sent_char, char2index).view(1, -1) for sent_char in c])
            pos = torch.cat([prepare_pos(sent_pos, pos2index).view(1, -1) for sent_pos in p])
            out += self.forward(inputs, char, pos).max(1)[1].data.tolist()
        return out
model = WindowClassifierWithCharEmb(len(word2index), len(char2index), len(pos2index), WORD_EMBEDDING_SIZE, CHAR_EMBEDDING_SIZE, POS_EMBEDDING_SIZE, WINDOW_SIZE, HIDDEN_SIZE, len(tag2index), BATCH_SIZE)
X_train, c_train, p_train, y_train = list(zip(*train_data))
X_test, c_test, p_test, y_test = list(zip(*test_data))

class_weight = compute_class_weight('balanced', tagset, y_train)

W = np.array([class_weight[tag2index[t] - 2] for t in y_train])

model.fit(X_train, c_train, p_train, y_train, W)
preds = []
trues = []

for test in test_data:
    x, c, p, y = test[0], test[1], test[2], test[3]
    input_ = prepare_word(x, word2index).view(1, -1)
    char_ = prepare_char(c, char2index).view(1, -1)
    pos_ = prepare_pos(p, pos2index).view(1, -1)
    i = model(input_, char_, pos_).max(1)[1]
    pred = index2tag[i.data.tolist()[0]]
    
    preds.append(pred)
    trues.append(y)

print(accuracy_score(preds, trues))
print(classification_report(preds, trues))
