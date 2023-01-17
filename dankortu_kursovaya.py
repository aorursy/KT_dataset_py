# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import torch

from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import TweetTokenizer



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/imdb_master.csv", encoding="ISO-8859-1")
print(data.groupby(['type']).count())
test_data = data[data.type == "test"]

train_data = data[data.type == "train"][data.label != "unsup"]
print(len(train_data))
import random

def foo(data):

    train_x = []

    train_y = []

    for i in range(len(data)):

        train_x.append([i.lower() for i in TweetTokenizer().tokenize(data.iloc[i].review) if i.isalpha()])

        label = data.iloc[i].label

        if label == "neg":

            train_y.append(0)

        elif label == "pos":

            train_y.append(1)

    return train_x, train_y
train_x, train_y = foo(train_data)
counter1 = 0

counter0 = 0

c = 0

for i in train_y:

    if i == 1:

        counter1 += 1

    else: counter0 += 1

    c +=1

print(counter0, counter1)
val_x = []

val_y = []

for i in range(3000):

    ind = random.randint(0, len(train_x)-1)

    val_x.append(train_x[ind].copy())

    val_y.append(train_y[ind])

    del train_x[ind]

    del train_y[ind]
c0 = 0

c1 = 0

for i in val_y:

    if i == 0:

        c0 += 1

    else: c1 += 1

print(c0, c1)
test_x, test_y =  foo(test_data)
from collections import Counter



counter = Counter()

for i in train_x:

    for j in i:

        counter[j] += 1

print(counter.most_common(3))
print(counter.most_common()[:5])
vocabular = {}

for num, i in enumerate(counter.most_common()):

    vocabular[i[0]] = num+1

    if num > 10000:

        break
print(len(vocabular))
def tokenize(x):

    train_tokenize_x = []

    for i in x:

        temp = []

        for j in i:

            tokenize = vocabular.get(j)

            if tokenize is not None:

                temp.append(tokenize)

        train_tokenize_x.append(temp)

    for i in range(len(train_tokenize_x)):

        if len(train_tokenize_x[i]) < 200:

            train_tokenize_x[i] += [0 for i in range(200-len(train_tokenize_x[i]))]

        else:

            train_tokenize_x[i] = train_tokenize_x[i][:200]

    return train_tokenize_x
train_tokenize_x = tokenize(train_x)

print(train_tokenize_x[3])
val_tokenize_x = tokenize(val_x)

print(val_tokenize_x[1])
test_tokenize_x = tokenize(test_x)

print(test_tokenize_x[0])
class Model(torch.nn.Module):

    def __init__(self, v_size):

        super().__init__()

        self.l1 = torch.nn.Embedding(v_size, 50, padding_idx=0)

        self.l2 = torch.nn.LSTM(50, 500, batch_first=True)

        self.l3 = torch.nn.Linear(500, 1)

        

    def forward(self, x):

        out = self.l1(x)

        out, (h_t, h_c) = self.l2(out)

        out = self.l3(h_t)

        return torch.sigmoid(out)
model = Model(len(vocabular)+2).cuda()
class Mydataset(Dataset):

    def __init__(self, x, y):

        self.x = x

        self.y = y

        

    def __getitem__(self, n):

        return torch.tensor(data=self.x[n]), torch.FloatTensor(data=[self.y[n]])

    

    def __len__(self):

        return len(self.x)
train_dataset = Mydataset(train_tokenize_x, train_y)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataset = Mydataset(val_tokenize_x, val_y)

val_dataloader = DataLoader(val_dataset, batch_size=20)
test_dataset = Mydataset(test_tokenize_x, test_y)

test_dataloader = DataLoader(test_dataset, batch_size=20)
optim = torch.optim.Adam(model.parameters(), lr=0.002)

criterion = torch.nn.BCELoss()
from sklearn.metrics import classification_report

for i in range(7):

    c = 0

    model.train()

    for x, y in train_dataloader:

        optim.zero_grad()

        x, y = x.cuda(), y.cuda()

        out = model(x)[0]

        loss = criterion(out, y)

        loss.backward()

        optim.step()

        if c % 50 == 0:

            print(c/len(train_dataloader))

        c += 1

    model.eval()

    y_labels = []

    y_predicted = []

    with torch.no_grad():

        c = 0

        for x, y in val_dataloader:

            y_labels.extend(y.tolist())

            x, y = x.cuda(), y.cuda()

            out = model(x)[0]

            out_value = []

            for i in out.tolist():

                if i[0] > 0.5:

                    out_value.append(1)

                else:

                    out_value.append(0)

            y_predicted.extend(out_value)

            if c % 50 == 0:

                print(c/len(val_dataloader))

            c += 1

    print(classification_report(y_labels, y_predicted))

        

    

    
from sklearn.metrics import classification_report

y_labels = []

y_predicted = []

with torch.no_grad():

    c = 0

    for x, y in test_dataloader:

        y_labels.extend(y.tolist())

        x, y = x.cuda(), y.cuda()

        out = model(x)[0]

        out_value = []

        for i in out.tolist():

            if i[0] > 0.5:

                out_value.append(1)

            else:

                out_value.append(0)

        y_predicted.extend(out_value)

        if c % 100 == 0:

            print(c/len(test_dataloader))

        c += 1

print(classification_report(y_labels, y_predicted))
from keras.preprocessing.text import Tokenizer

df = pd.read_csv('../input/imdb_master.csv',encoding="latin-1")

df = df.drop(['Unnamed: 0','file'],axis=1)

df.columns = ['type',"review","sentiment"]

df.head()

print(df.head())

df = df[df.sentiment != 'unsup']

df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})



df_train = df[df.type == 'train']

df_test = df[df.type == 'test']

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

def ngram_vectorize(train_texts, train_labels, val_texts):

    kwargs = {

        'ngram_range' : (1, 2),

        'dtype' : 'int32',

        'strip_accents' : 'unicode',

        'decode_error' : 'replace',

        'analyzer' : 'word',

        'min_df' : 2,

    }

    

    tfidf_vectorizer = TfidfVectorizer(**kwargs)

    x_train = tfidf_vectorizer.fit_transform(train_texts)

    x_val = tfidf_vectorizer.transform(val_texts)

    

    selector = SelectKBest(f_classif, k=min(6000, x_train.shape[1]))

    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype('float32')

    x_val = selector.transform(x_val).astype('float32')

    return x_train, x_val



df_bag_train, df_bag_test = ngram_vectorize(df_test['review'], df_test['sentiment'], df_train['review'])

df_bag_train, df_bag_test = ngram_vectorize(df_test['review'], df_test['sentiment'], df_train['review'])

from sklearn import metrics



nb = MultinomialNB()

nb.fit(df_bag_train, df_train['sentiment'])

nb_pred = nb.predict(df_bag_test)

print(metrics.classification_report(df_test['sentiment'], nb_pred))

cm = metrics.confusion_matrix(nb_pred, df_test['sentiment'])

print('Accuracy ',metrics.accuracy_score(df_test['sentiment'], nb_pred))