!pip install youtokentome
import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from tqdm import tqdm, trange  # VERY useful progress bar

from sklearn.model_selection import train_test_split

import time 

import math

import pandas as pd

import youtokentome as yttm



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



max_length = 100

batch_size = 256

freq_thresh = 32

n_hidden = 256

embedding_size = 256

n_cats = 50
train = pd.read_csv('../input/text-classification-useful-files/train_preprocessed.csv')

test = pd.read_csv('../input/text-classification-useful-files/test_preprocessed.csv')



train.fillna('', inplace=True)

test.fillna('', inplace=True)
# this part is used for text preprocessing and should be exectued only once

# preprocessing here is actually incomplete, it preserves almost all words in text

# removing punctuation supposedly was a bad idea



# import nltk

# nltk.download("stopwords") # was not the wise choice



# from nltk.corpus import stopwords

# from pymystem3 import Mystem

# from string import punctuation



# mystem = Mystem() 

# stopwords = stopwords.words('russian')



# def preprocess_text(text):

#     tokens = mystem.lemmatize(text.lower())

#     tokens = [token for token in tokens if token not in stopwords\

#               and token != " " \

#               and token.strip() not in punctuation]

    

#     text = " ".join(tokens)

    

#     return text



# train['title&description'] = train['title&description'].progress_apply(preprocess_text)

# test['title&description'] = test['title&description'].progress_apply(preprocess_text)



# train.head()
# category mapping to 0-n_cats-1 range



cats_dict = dict(zip(np.unique(train['Category']), range(n_cats)))

rev_cats_dict = {v: k for k, v in cats_dict.items()}



train['Category'] = train['Category'].map(cats_dict)



train.head()
# Bag-of-words embedding, was actually used to generate the dictionary and leave only most common words

# due to the fact that embedding vectors for least common words would not be updated frequently

# and the size of the layer would be too big. Later this approach was discarded in favour of Byte Pair Encoding.



# words = {} 



# for text in tqdm(train['title&description']):

#     for word in text.split(' '):

#         if word not in words:

#             words[word] = 1

#         else:

#             words[word] += 1

            

# for text in tqdm(test['title&description']):

#     for word in text.split(' '):

#         if word not in words:

#             words[word] = 1

#         else:

#             words[word] += 1

            

# for word in tqdm(list(words.keys())):

#     if words[word] < freq_thresh:

#         del words[word]

        

# for i, word in enumerate(words):

#     words[word] = i + 2

        

# print(len(words))
# loading Byte Pair Encoding model, trained on text corpus from train in separate file



bpe = yttm.BPE(model='../input/text-classification-useful-files/bpe.model')

vocab = bpe.vocab()



words = dict(zip(vocab, range(len(vocab))))
# Architecture: Embedding + LSTM

# num_layers=2 in LSTM actually means two stacked LSTMS

# bidirectional=True means that LSTM reads sequence both ways: first to last and last to first 

# heavy Dropout regularization improved performance 

# SpatialDropout removes a subset of features from every embedding vector respectively, not independetly like

# in regular dropout 



class RNN(nn.Module):

    def __init__(self, hidden_size):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size



        self.embedding = nn.Embedding(len(words) + 2 , embedding_size, padding_idx=0)

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)

        #self.out1 = nn.Linear(hidden_size * 2, hidden_size) # better performance without additional fc layer

        self.dropout = nn.Dropout(0.5)

        self.spatial_dropout = nn.Dropout2d(0.2)

        self.out = nn.Linear(hidden_size * 2, n_cats)

        self.softmax = nn.LogSoftmax(dim=1)

        

    def forward(self, inp, hidden):

        batch_size, seq_len = inp.size()

        

        embeds = self.embedding(inp)

        embeds = embeds.permute(0, 2, 1)   # convert to [batch, channels, time]

        embeds = self.spatial_dropout(embeds)

        embeds = embeds.permute(0, 2, 1)   # back to [batch, time, channels]

        

        output, hidden = self.lstm(embeds.view(batch_size, seq_len, -1), hidden)

        

        output = output.permute([1, 0, 2])

        #output = F.relu(self.dropout(self.out1(output[-1]))) # better performance without additional fc layer

        output = self.softmax(self.out(output[-1]))

        return output



    def initHidden(self):

        return (torch.zeros(4, batch_size, self.hidden_size, device=device), torch.zeros(4, batch_size, self.hidden_size, device=device))

    

model = RNN(n_hidden).to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5) # essential technique even for Adam!



criterion = nn.NLLLoss() # == CrossEntropyLoss, we just use activated inputs here instead of raw
# bag-of-word version of sequence preparation

# def prepare_sequence(seq):

#     res = np.zeros(max_length)

#     idxs = np.array([words[w] if w in words else 1 for w in seq.split(' ')])

    

#     if idxs.shape[0] > max_length:

#         idxs = idxs[:max_length]

        

#     res[:idxs.shape[0]] = idxs

#     return res



# we truncate sequences to max_length for easier use of batched training 

def prepare_sequence(seq):

    res = np.zeros(max_length, dtype='int32')

    idxs = np.array(bpe.encode(seq, output_type=yttm.OutputType.ID), dtype='int32')

    

    if idxs.shape[0] > max_length:

        idxs = idxs[:max_length]

        

    res[:idxs.shape[0]] = idxs

    return res



def train_iteration(inp, target):

    criterion = nn.NLLLoss()

    hidden = model.initHidden()

    

    optimizer.zero_grad()

    

    y_pred = model(inp, hidden)

    

    loss = criterion(y_pred, target)

    loss.backward()



    optimizer.step()



    return loss.item()
# PyTorch doesn't provide progress bar unlike Keras, so we use some custom functions to track time



def asMinutes(s):

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)





def timeSince(since, percent):

    now = time.time()

    s = now - since

    es = s / (percent)

    rs = es - s

    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(train['title&description'], train['Category'], 

                                                      test_size = 0.1, random_state=42)



X_train = np.array([prepare_sequence(seq) for seq in tqdm(X_train)])

X_valid = np.array([prepare_sequence(seq) for seq in tqdm(X_valid)])

y_train = np.array(y_train)

y_valid = np.array(y_valid)



def trainIters(model, batch_size=batch_size, print_every=1000, learning_rate=0.001):

    model.train() # don't forget this!

    

    start = time.time()

    plot_losses = []

    print_loss_total = 0  

    plot_loss_total = 0  

    

    # we don't use 'fair' shuffle of whole train array before each epoch, we use random selection instead 

    # for memory efficiency (copy of train array didn't fit in my RAM)

    for iter in range(1, (len(X_train) + 1) // batch_size):

        i = np.random.randint(len(y_train) - 1, size=batch_size)

        input_tensor = torch.LongTensor(X_train[i])

        target_tensor = torch.LongTensor(y_train[i])

        

        loss = train_iteration(input_tensor.to(device), target_tensor.to(device))

        

        print_loss_total += loss

        plot_loss_total += loss



        if iter % print_every == 0:

            print_loss_avg = print_loss_total / print_every

            print_loss_total = 0

            print('%s (%d iters, %d%%) %.4f' % (timeSince(start, iter * batch_size / len(X_train)),

                                         iter * batch_size, iter * batch_size / len(X_train) * 100, print_loss_avg))

            

def evaluate(model):

    model.eval() # don't forget this!

    

    loss = 0.0

    acc = 0.0 

    

    with torch.no_grad():

        for i in range(len(X_valid) // batch_size):

            i = range(i * batch_size, (i + 1) * batch_size)

            input_tensor = torch.LongTensor(X_valid[i]).to(device)

            target_tensor = torch.LongTensor(y_valid[i]).to(device)

            

            hidden = model.initHidden()

            

            y_pred = model(input_tensor, hidden)

            

            loss += criterion(y_pred, target_tensor)

            acc += (torch.argmax(y_pred, dim=1) == target_tensor).float().sum()

    

    loss /= (len(X_valid) // batch_size)

    acc /= (len(X_valid) // batch_size) * batch_size

    

    return loss, acc
# tracked train array memory consumption (in GB)

print(X_train.nbytes / 1024 / 1024 / 1024 + X_valid.nbytes / 1024 / 1024 / 1024 )
# checkpoint load to resume training 



#checkpoint = torch.load('state_last')



#model.load_state_dict(checkpoint['model_state_dict'])

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
n_epochs = 16 # I used up to 20



for i in range(n_epochs):

    print("Epoch %d" % (i + 1))

    trainIters(model)

    if (i + 1) % 3 == 0:

        lr_scheduler.step() # reduce learning rate for better convergence

    print('Val loss: {:4f}\nVal acc: {:4f}'.format(*evaluate(model)))
# save model checkpoint to resume training 



#torch.save({'model_state_dict': model.state_dict(),

#            'optimizer_state_dict': optimizer.state_dict()

#            }, 'state_last')
print('Val loss: {:4f}\nVal acc: {:4f}'.format(*evaluate(model)))
# answer csv generation



import pandas as pd



ans = pd.DataFrame(columns=['Id', 'Category'])

ans['Id'] = test['itemid']
X_test = [prepare_sequence(seq) for seq in tqdm(test['title&description'])]



# this is required for batched inference, we will later drop predictions on padded objects

pad = len(X_test) % batch_size  



for i in range(batch_size - pad):

    X_test.append(np.zeros(max_length))



X_test = np.array(X_test)

print(X_test.shape[0] % batch_size)
test_y = []



model.eval()

with torch.no_grad():

    for i in trange(len(X_test) // batch_size):

        i = range(i * batch_size, (i + 1) * batch_size)

        input_tensor = torch.LongTensor(X_test[i]).to(device)

        

        hidden = model.initHidden()

        

        y_pred = model(input_tensor, hidden)

        

        for cat in torch.argmax(y_pred, dim=1).cpu().detach().numpy():

            test_y.append(rev_cats_dict[cat])
# just to make sure we have matching lengths



print(len(test_y) - (batch_size - pad))

print(len(ans['Id']))
ans['Category'] = test_y[:-(batch_size - pad)]
# I prefer to encode a brief description of the approach the csv was generated, it helps with blending afterwards

# and generally helps to keep track of what worked and what didn't

ans.to_csv("lstm_even_bigger_preprocessed_bpe_less_agressive_lr_decay_8.5_epochs.csv",index=False)