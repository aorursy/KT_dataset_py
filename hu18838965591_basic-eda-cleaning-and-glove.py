import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim

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

train= pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
train.head(3)
print('There are {} rows and {} columns in train'.format(train.shape[0],train.shape[1]))
print('There are {} rows and {} columns in train'.format(test.shape[0],test.shape[1]))
x=train.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')
def create_corpus(target):
    corpus=[]
    
    for x in train[train['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus
corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

x,y=zip(*top)
plt.bar(x,y)
corpus=create_corpus(1)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    


x,y=zip(*top)
plt.bar(x,y)
plt.figure(figsize=(10,5))
corpus=create_corpus(1)

dic=defaultdict(int)
import string
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y=zip(*dic.items())
plt.bar(x,y)
plt.figure(figsize=(10,5))
corpus=create_corpus(0)

dic=defaultdict(int)
import string
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y=zip(*dic.items())
plt.bar(x,y,color='green')

counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:40]:
    if (word not in stop) :
        x.append(word)
        y.append(count)
sns.barplot(x=y,y=x)
def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
plt.figure(figsize=(10,5))
top_tweet_bigrams=get_top_tweet_bigrams(train['text'])[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x)
print (train.shape, test.shape)
df=pd.concat([train,test])
df.shape
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

remove_URL(example)
df['text']=df['text'].apply(lambda x : remove_URL(x))
example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
print(remove_html(example))
df['text']=df['text'].apply(lambda x : remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

remove_emoji("Omg another Earthquake 😔😔")
df['text']=df['text'].apply(lambda x: remove_emoji(x))

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

example="I am a #king"
print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
!pip install pyspellchecker
from spellchecker import SpellChecker
correct_cnt = 0
spell = SpellChecker()
def correct_spellings(text):
    global correct_cnt
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
            correct_cnt += 1
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
        
text = "corect me plese"
correct_spellings(text)
df['text_new']=df['text'].apply(lambda x : correct_spellings(x))
print (correct_cnt)
embedding_dict={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()
print (df.columns)
print (df.head(3))
def build_vocab(X):
    
    tweets = X.apply(lambda s: [word.lower() for word in word_tokenize(s) if((word.isalpha()==1) & (word not in stop))]).values      
    vocab = {}
    
    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1                
    return vocab


def check_embeddings_coverage(X, embeddings):
    
    vocab = build_vocab(X)    
    
    covered = {}
    oov = {}    
    n_covered = 0
    n_oov = 0
    
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
            
    vocab_coverage = len(covered) / len(vocab)
    text_coverage = (n_covered / (n_covered + n_oov))
    
    return vocab_coverage, text_coverage

df_train = df[~df['target'].isna()]
df_test = df[df['target'].isna()]

train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(df_train['text'], embedding_dict)
test_glove_vocab_coverage, test_glove_text_coverage = check_embeddings_coverage(df_test['text'], embedding_dict)
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))
xtrain, xvalid, ytrain, yvalid = train_test_split(df_train.text.values, df_train.target.values, 
                                                  stratify = df_train.target.values, 
                                                  random_state = 2020, 
                                                  test_size = 0.3, shuffle = True)
xtest = df_test.text.values
ytest = df_test.target.values
print (xtrain.shape)
print (xvalid.shape)
print (xtest.shape)
def sent2vec(s):
    words = [word.lower() for word in word_tokenize(str(s)) if((word.isalpha()==1) & (word not in stop_words))]
    M = []
    for w in words:
        w = str(w)
        try:
            torch_tmp = list(embedding_dict[w])
            M.append(torch_tmp)
        except:
            continue
    if len(M) == 0:
        M.append([0] * 100)
    return M

xtrain_glove = [sent2vec(x) for x in xtrain]
xvalid_glove = [sent2vec(x) for x in xvalid]
xtest_glove = [sent2vec(x) for x in xtest]
word_vector_size = len(xtrain_glove[0][0])
label_size = 2
print ('word_vector_size: {:}, label_size: {:}'.format(word_vector_size, label_size))

xtrain_lengths = torch.LongTensor([len(x) for x in xtrain_glove]).cuda()
print ('xtrain_lengths_max: {:}'.format(xtrain_lengths.max()))
max_length = int(xtrain_lengths.max())
xtrain_torch = torch.zeros((len(xtrain_glove), max_length, word_vector_size)).float().cuda()
for idx in range(len(xtrain_glove)):
    seqlen = min(int(xtrain_lengths[idx].cpu().numpy()), max_length)
    xtrain_torch[idx, :seqlen] = torch.FloatTensor(np.array(xtrain_glove[idx])[: seqlen, :])

print (type(xtrain_torch), xtrain_torch.size())
xtrain_lengths, seq_idx = xtrain_lengths.sort(0, descending = True)
xtrain_torch = xtrain_torch[seq_idx]
if isinstance(ytrain, np.ndarray):
    ytrain = torch.from_numpy(ytrain).cuda()[seq_idx].long()
else:
    ytrain = ytrain[seq_idx].long()
xvalid_lengths = torch.FloatTensor([len(x) for x in xvalid_glove]).cuda()
xvalid_torch = torch.zeros((len(xvalid_glove), max_length, word_vector_size)).float().cuda()
for idx in range(len(xvalid_glove)):
    seqlen = min(int(xvalid_lengths[idx].cpu().numpy()), max_length)
    xvalid_torch[idx, :seqlen] = torch.FloatTensor(np.array(xvalid_glove[idx])[: seqlen, :])

print (type(xvalid_torch), xvalid_torch.size())
xvalid_lengths, seq_idx_valid = xvalid_lengths.sort(0, descending = True)
xvalid_torch = xvalid_torch[seq_idx_valid]
if isinstance(yvalid, np.ndarray):
    yvalid = torch.from_numpy(yvalid).cuda()[seq_idx_valid].long()
else:
    yvalid = yvalid[seq_idx_valid].long()

def get_val_score():
    # See what the scores are after training
    with torch.no_grad():
        xvalid_length_batch = torch.clamp(xvalid_lengths, 0, 100).cuda()
    #     print (type(xvalid_length_batch), xvalid_length_batch.device, xvalid_length_batch.size())
    #     print (type(xvalid_torch), xvalid_torch.device, xvalid_torch.size())
        embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(xvalid_torch, xvalid_length_batch, batch_first=True)
        outputs = model(embed_input_x_packed, xvalid_torch.size(0))
        _, predicted = torch.max(outputs.data, 1)

        total = xvalid_torch.size(0)

        TP = (predicted == yvalid).sum().item()
        FP = ((predicted == 1) & (yvalid == 0)).sum().item()
        FN = ((predicted == 0) & (yvalid == 1)).sum().item()
        acc = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * acc * recall / (acc + recall)

    print('Accuracy: {} / {} = {:.2f}%, Recall: {} / {} = {:.2f}%, f1_score: {:.4f}'. \
              format(TP, (TP + FP), (100 * TP / (TP + FP)), TP, (TP + FN), (100 * TP / (TP + FN)), f1))
get_val_score()
input_size = word_vector_size
num_classes = 2
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
epoch_size = 25
for epoch in range(epoch_size):  # again, normally you would NOT do 300 epochs, it is toy data
    total = 0
    TP = 0
    FP = 0
    FN = 0
    for i, (xtrain_batch, ytrain_batch, xtrain_length_batch) in enumerate(train_loader):
        xtrain_length_batch = torch.clamp(xtrain_length_batch, 0, 100)
        # print (xtrain_batch.size(), xtrain_length_batch.size())
        embed_input_x_packed = nn.utils.rnn.pack_padded_sequence(xtrain_batch, xtrain_length_batch, batch_first=True)
        
        # forward pass
        outputs = model(embed_input_x_packed, len(xtrain_batch))
        # print ("outputs.size(): {:}".format(outputs.size()))
        
        _, predicted = torch.max(outputs.data, 1)
        total += xtrain_batch.size(0)
        TP += (predicted == ytrain_batch).sum().item()
        FP += ((predicted == 1) & (ytrain_batch == 0)).sum().item()
        FN += ((predicted == 0) & (ytrain_batch == 1)).sum().item()
        
        loss = criterion(outputs, ytrain_batch)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         if epoch % 20 == 0 and i == 0:
#             print (i, correct, total)
#             print (Counter(predicted.cpu().numpy()), Counter(ytrain_batch.cpu().numpy()),)
    if (epoch + 1) % 3 == 0:
        acc = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * acc * recall / (acc + recall)
        print ('\n')
        print('Epoch [{}/{}], Accuracy: {} / {} = {:.2f}%, Recall: {} / {} = {:.2f}%, f1_score: {:.4f},  Loss: {:.4f}, time: {:.1f}'. \
              format(epoch + 1, epoch_size, TP, (TP + FP), (100 * TP / (TP + FP)), 
                    TP, (TP + FN), (100 * TP / (TP + FN)), 
                     f1, 
                    loss.item(), time.time() - start_time))
        get_val_score()
!mkdir /kaggle/working/nlp-getting-started
df_sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
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
    _, predicted = torch.max(outputs.data, 1)

print (Counter(predicted.cpu().numpy()))

test_id = df_sub['id'].values
test_id = test_id[list(seq_idx_test.cpu().numpy())]
df_sub_test = pd.DataFrame(data = predicted.cpu().numpy(), columns = ['score'])
df_sub_test['id'] = test_id
df_sub_test['target'] = df_sub_test['score'].apply(lambda x: 1 if x >= 0.5 else 0)
df_sub_test = df_sub_test[['id', 'target']]
df_sub_test.to_csv('/kaggle/working/nlp-getting-started/submission_20201009_02.csv', index = False)
print (df_sub_test.head(3))
