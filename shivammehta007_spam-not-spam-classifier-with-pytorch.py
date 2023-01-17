import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import torchvision

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

import os



import nltk

from nltk.tokenize import word_tokenize

from nltk import PorterStemmer

from nltk import WordNetLemmatizer

from nltk.corpus import stopwords



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
os.listdir('../input/')
data = pd.read_csv('../input/spam_or_not_spam.csv')

data.head()
data.dropna(inplace=True)

change_labels = lambda x: 1 if x==0 else 0

data['label'] = data['label'].apply(change_labels)

data.head()
remove_non_alphabets =lambda x: re.sub(r'[^a-zA-Z]',' ',x)
tokenize = lambda x: word_tokenize(x)
ps = PorterStemmer()

stem = lambda w: [ ps.stem(x) for x in w ]
lemmatizer = WordNetLemmatizer()

leammtizer = lambda x: [ lemmatizer.lemmatize(word) for word in x ]
print('Processing : [=', end='')

data['email'] = data['email'].apply(remove_non_alphabets)

print('=', end='')

data['email'] = data['email'].apply(tokenize) # [ word_tokenize(row) for row in data['email']]

print('=', end='')

data['email'] = data['email'].apply(stem)

print('=', end='')

data['email'] = data['email'].apply(leammtizer)

print('=', end='')

data['email'] = data['email'].apply(lambda x: ' '.join(x))

print('] : Completed', end='')

data.head()
max_words = 10000

cv = CountVectorizer(max_features=max_words, stop_words='english')

sparse_matrix = cv.fit_transform(data['email']).toarray()
sparse_matrix.shape
x_train, x_test, y_train, y_test = train_test_split(sparse_matrix, np.array(data['label']))
class LogisticRegression(nn.Module):

    def __init__(self):

        super(LogisticRegression, self).__init__()

        self.linear1 = nn.Linear(10000, 100)

        self.linear2 = nn.Linear(100, 10)

        self.linear3 = nn.Linear(10, 2)

        

    def forward(self, x):

        x = F.relu(self.linear1(x))

        x = F.relu(self.linear2(x))

        x = self.linear3(x)

        return x
model = LogisticRegression()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters() , lr=0.01)
x_train = Variable(torch.from_numpy(x_train)).float()

y_train = Variable(torch.from_numpy(y_train)).long()
epochs = 20

model.train()

loss_values = []

for epoch in range(epochs):

    optimizer.zero_grad()

    y_pred = model(x_train)

    loss = criterion(y_pred, y_train)

    loss_values.append(loss.item())

    pred = torch.max(y_pred, 1)[1].eq(y_train).sum()

    acc = pred * 100.0 / len(x_train)

    print('Epoch: {}, Loss: {}, Accuracy: {}%'.format(epoch+1, loss.item(), acc.numpy()))

    loss.backward()

    optimizer.step()

plt.plot(loss_values)

plt.title('Loss Value vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Loss'])

plt.show()
x_test = Variable(torch.from_numpy(x_test)).float()

y_test = Variable(torch.from_numpy(y_test)).long()
model.eval()

with torch.no_grad():

    y_pred = model(x_test)

    loss = criterion(y_pred, y_test)

    pred = torch.max(y_pred, 1)[1].eq(y_test).sum()

    print ("Accuracy : {}%".format(100*pred/len(x_test)))
