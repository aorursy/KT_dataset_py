import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import torch.nn.functional as F
data = pd.read_csv("../input/SMSSpamCollection.tsv", delimiter='\t', header=None, names=["outcome", 'message'])
data.head()
data.outcome = data.outcome.map({'ham':0, 'spam':1})
data.head()
features = data.message.values
labels = data.outcome.values
num_words = 1000
features.shape
labels.shape
t = Tokenizer(num_words=1000)
t.fit_on_texts(features)
features = t.texts_to_matrix(features, mode='tfidf')
features.shape
# Building model
class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden , hidden)
        self.l3 = nn.Linear(hidden, 2)
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out        
input = 1000
hidden=100
output = 2
model = Model(input, hidden, output)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, shuffle=True, random_state=34)
# params
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
def train(epochs):
    x_train = Variable(torch.from_numpy(features_train)).float()
    y_train = Variable(torch.from_numpy(labels_train)).long()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        print ("epoch #",epoch)
        print ("loss: ", loss.item())
        pred = torch.max(y_pred, 1)[1].eq(y_train).sum()
        print ("acc:(%) ", 100*pred/len(x_train))
        loss.backward()
        optimizer.step()
def test(epochs):
    model.eval()
    x_test = Variable(torch.from_numpy(features_test)).float()
    y_test = Variable(torch.from_numpy(labels_test)).long()
    for epoch in range(epochs):
        with torch.no_grad():
            y_pred = model(x_test)
            loss = criterion(y_pred, y_test)
            print ("epoch #",epoch)
            print ("loss: ", loss.item())
            pred = torch.max(y_pred, 1)[1].eq(y_test).sum()
            print ("acc (%): ", 100*pred/len(x_test))

train(100)
test(100)
pred = model(torch.from_numpy(features_test).float())
pred
pred = torch.max(pred,1)[1]
len(pred)
len(features_test)
pred = pred.data.numpy()
pred
labels_test
accuracy_score(labels_test, pred)
p_train = model(torch.from_numpy(features_train).float())
p_train = torch.max(p_train,1)[1]
len(p_train)
p_train = p_train.data.numpy()
p_train
accuracy_score(labels_train, p_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, pred)
print (cm)
