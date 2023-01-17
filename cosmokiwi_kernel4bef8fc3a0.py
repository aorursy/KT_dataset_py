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
from tqdm import tqdm_notebook

from nltk.tokenize import TweetTokenizer

from torch.utils.data import DataLoader, Dataset

import torch.nn as nn

import torch

import torch.optim as optim

from sklearn.metrics import classification_report
np.random.seed(42)

# Максимальное количество слов (по частоте использования)

max_features = 5000

# Максимальная длина рецензии в словах

max_len = 110
data = "../input/aclimdb/aclImdb/"

train_data = data + "train/"

test_data = data + "test/"
print(os.listdir(test_data))

print(os.listdir(train_data))
with open(data + "imdb.vocab") as f:

    voc = {}

    counter = 0

    for i in f.read().split():

        if (counter < max_features):

            voc[i] = counter

            counter += 1

voc_size = len(voc)



print(voc_size)
train_text = []

test_text = []

train_label = []

test_label = []

valid_text = []

valid_label = []





for train_test in ['train','test']:

    for neg_pos in ['neg','pos']:

        file_path = data + train_test + '/' + neg_pos + '/'

        for file in tqdm_notebook(os.listdir(file_path)):

            with open(file_path + file) as f:

                text = TweetTokenizer().tokenize(f.read().lower()) 

                text_token = []

                for i in text:

                    d = voc.get(i)

                    if d is not None:

                        text_token.append(d)

                        

                if len(text_token)>max_len:

                    text_token = text_token[:max_len]

                else:

                    for i in range(max_len - len(text_token)):

                        text_token.append(0)

                        

                if train_test == 'train':

                    train_text.append(text_token)

                    if neg_pos == 'neg':

                        train_label.append(0)

                    else:

                        train_label.append(1)

                else:

                    test_text.append(text_token)

                    if neg_pos == 'neg':

                        test_label.append(0)

                    else:

                        test_label.append(1)
print (train_text[2])
X_train = pd.DataFrame()

X_train['review'] = train_text

X_train['label'] = train_label



X_test = pd.DataFrame()

X_test['review'] = test_text

X_test['label'] = test_label
X_train.head()
class MyDataset(Dataset):

    def __init__(self, review, label):

        self.review = review

        self.label  = label

        

    def __len__(self):

        return len(self.review)

    

    def __getitem__(self, i):

        review = torch.tensor(data = self.review[i])

        label = torch.FloatTensor(data = [self.label[i]])

        return review, label
print(len(test_text))
valid_text = test_text[11500:13500]

valid_label = test_label[11500:13500]



#for i in range (len(test_text))
print(type(test_text))
train_dataset = MyDataset(train_text, train_label)

valid_dataset = MyDataset(valid_text, valid_label)

test_dataset = MyDataset(test_text, test_label)



train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=15)

test_loader = DataLoader(test_dataset, batch_size=15) 
class RecNN(nn.Module):

    def __init__(self, voc_size, embedding_size, hidden_size):

        super().__init__()

        self.embedding = nn.Embedding(voc_size, embedding_size, padding_idx=0)

        self.lstm1 = nn.LSTM(embedding_size, hidden_size, batch_first=True)

        self.lin  = nn.Linear(hidden_size, 1)

        

         

    def forward(self, x):

        x = self.embedding(x)

        x, (h_t, h_c) = self.lstm1(x)

        out = self.lin(h_t)

        return torch.sigmoid(out)

        
model = RecNN(voc_size, 38, 274).cuda()

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epoches = 5
for epoche in range(epoches):

    model.train()

    for c,(xx, yy) in tqdm_notebook(enumerate(train_loader), total=len(train_loader)):

        xx, yy = xx.cuda(), yy.cuda()

        optimizer.zero_grad()

        out = model(xx)[0]

        loss = criterion(out, yy)

        loss.backward()

        optimizer.step()



        if c % 300 == 0:

            print(epoche, loss.item())

            

    model.eval()

    y_test = []

    y_pred = []



    with torch.no_grad():

        for c,(xx, yy) in tqdm_notebook(enumerate(valid_loader), total=len(valid_loader)):

            xx, yy = xx.cuda(), yy.cuda()

            out = model(xx)[0]

            predicted = [i[0]>0.5 for i in out.tolist()]

            y_pred.extend(predicted)

            y_test.extend(yy.tolist())

            

            if (c% 300 == 0):

                print (c, len(valid_loader))

        

    print(classification_report(y_test, y_pred)) 
model.eval()

y_test = []

y_pred = []



with torch.no_grad():

    for c,(xx, yy) in tqdm_notebook(enumerate(test_loader), total=len(test_loader)):

        xx, yy = xx.cuda(), yy.cuda()

        out = model(xx)[0]

        predicted = [i[0]>0.5 for i in out.tolist()]

        y_pred.extend(predicted)

        y_test.extend(yy.tolist())

            

        if (c% 300 == 0):

            print (c, len(test_loader))

        

print(classification_report(y_test, y_pred)) 