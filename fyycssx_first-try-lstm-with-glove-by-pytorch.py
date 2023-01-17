# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

import string
#load the datasets

tweets = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#have a look at data

tweets
#write function for data clean

#all codes bellow in this cell is from https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove#Data-Cleaning

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

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

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)    
#make the data clean functions a pipeline 

remove_funcs = [remove_emoji,remove_URL,remove_punct,remove_html]

def clean_text(text,funcs):

    text = text.lower()

    for func in funcs:

        text = func(text)

    return text



sample = tweets['text'].values[100]

print(sample)

print(clean_text(sample,remove_funcs))

clean = lambda text:clean_text(text,remove_funcs)
#build the vocab

from nltk import word_tokenize

from functools import reduce

from collections import defaultdict,Counter

texts = tweets['text'].values.tolist() +  test['text'].values.tolist()

cleaned_texts = [clean(text) for text in texts]

word_counter = Counter()



for text in cleaned_texts:

    words = list(word_tokenize(text))

    word_counter.update(words)

print("word num:",len(word_counter))



word2index = {'unk':0}

for i,word in enumerate(word_counter.keys()):

    word2index[word] = i+1

index2word = {v:k for k,v in word2index.items()}
#load the glove embeddings

embeddings_dictionary = dict()

glove_file = open('/kaggle/input/glove6b50dtxt/glove.6B.50d.txt', encoding="utf8")

for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = np.asarray(records[1:], dtype='float32')

    embeddings_dictionary [word] = vector_dimensions

glove_file.close()
#prepare the embedings for lstm model

import torch

embedding_dim = 50

embedding_matrix = torch.zeros((len(word_counter)+1, embedding_dim))

for word, index in word2index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = torch.from_numpy(embedding_vector)
#prepare dataloader for lstm model

from torch.utils.data import Dataset,DataLoader

class TextDataSet(Dataset):

    def __init__(self, texts, labels=None):

        self.texts = texts

        self.labels = labels



    def __len__(self):

        return len(self.texts)



    def __getitem__(self, index):

        text = self.texts[index]

        if self.labels is not None:

            label = self.labels[index]

            return text, label

        else:

            return text

        

def texts2tensor(texts,word2index,pad_token = 0,max_len = 50):

    indexes_list = [[word2index.get(word,0) for word in word_tokenize(text)] for text in texts]

    

    max_len = min(max_len,max([len(indexes) for indexes in indexes_list]))

    if max_len > 50:

        raise Exception("max > 50")

    truncated_indexes = [indexes[:max_len] for indexes in indexes_list]

    padded_indexes = [indexes+[0]*(max_len - len(indexes)) for indexes in truncated_indexes]

    return torch.LongTensor(padded_indexes)



def train_collate(batch_inputs):

    texts,labels = zip(*batch_inputs)

    input_tensor = texts2tensor(texts,word2index)

    return input_tensor,torch.LongTensor(labels)



train_dataset = TextDataSet(tweets['text'].values.tolist(),tweets['target'].values.tolist())    

test_dataset = TextDataSet(test['text'])       

    

train_loader = DataLoader(train_dataset,batch_size= 20, shuffle = True,collate_fn=train_collate)

test_loader = DataLoader(test_dataset,batch_size=10,shuffle=False,collate_fn=lambda texts:texts2tensor(texts,word2index))
#define the lstm model for classification

from torch import nn

class LstmClassification(nn.Module):

    def __init__(self,embed_size,embedding,hidden_size=128):

        super(LstmClassification,self).__init__()

        self.embed_size = embed_size

        self.linear  = nn.Linear(in_features=hidden_size*2,out_features=2)

        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=2,bidirectional=True,batch_first=True)

        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])

        self.embedding.weight.data.copy_(embedding)

    

    def forward(self,inputs):

        embedded = self.embedding(inputs)

        outputs,(hs,cs) = self.lstm(embedded)

        return  self.linear(outputs[:,-1,:])
#define the loss function

loss_func = nn.CrossEntropyLoss()
#train the model

model = LstmClassification(embed_size=50,embedding = embedding_matrix)

epochs  = 20

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for e in range(1,epochs+1):

    num = 0

    total_batch = len(train_loader)

    epoch_loss = 0

    for inputs,labels in train_loader:

        num+=1

        

        logits = model(inputs)

        optimizer.zero_grad()

        loss = loss_func(logits,labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        print("\repoch:%d %d/%d loss:"%(e,num,total_batch),loss.item(),end="")

    print(" mean loss:",epoch_loss/total_batch)