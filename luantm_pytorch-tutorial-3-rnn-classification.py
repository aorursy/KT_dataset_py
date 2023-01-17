import torch
import os
import pandas as pd
import numpy as np
print(os.listdir('../input'))
import re
# import and explore data

train_data = pd.read_csv('../input/names_train.csv', names=['name', 'nation'])
test_data = pd.read_csv('../input/names_test.csv', names=['name', 'nation'])
print(train_data.head())
print('-'*40)
print(test_data.tail())

print('*'*40)

x_train = train_data['name']
y_train = train_data['nation']

x_test = test_data['name']
y_test = test_data['nation']

classes = list(set(y_train))
print(classes)

print('-' * 40)

name_max_len = 0
for name in x_train:
    if len(name) > name_max_len:
        name_max_len = len(name)
for name in x_test:
    if len(name) > name_max_len:
        name_max_len = len(name)

print(f'max len name: {name_max_len}')
print(f'len of nations: {len(classes)}')
        
x_train = [name.lower() for name in x_train]
x_test = [name.lower() for name in x_test]

y_train = [classes.index(nation) for nation in y_train]
y_test = [classes.index(nation) for nation in y_test]

print(x_train[0:20])
print(y_train[0:20])


chars = [chr(n) for n in range(97, 123)]

print('-'*40)
print(chars)
# init parameters
input_size = len(chars)
hidden_size = 18

seq_len = 20
batch_size = 100
num_layers = 1
# create model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
    def forward(self, name):
        # create init hidden
        hidden = self.init_hidden()
        # create data
        x = self.init_data(name)
        # RNN
        # input : (batch_size, seq_len, input_size)
        # hidden: (num_layers * num_direction, batch_size, hidden_size)
        output, hidden = self.rnn(x, hidden)
        return output, hidden.view(-1, hidden_size)
    
    def init_hidden(self):
        return torch.autograd.Variable(torch.zeros(num_layers, batch_size, hidden_size))
    def init_data(self, names):
        names = [re.sub('\W+','', name).lower() for name in names]
        x = torch.zeros(batch_size, seq_len, input_size)
        for batch in range(batch_size):
            for i, c in enumerate(names[batch]):
                idx = chars.index(c)
                x[batch][seq_len - i - 1][idx] = 1
        
        return torch.autograd.Variable(x)
    
model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)
# train
for epoch in range(10000):
    optimizer.zero_grad()
    loss = 0
    count = 0
    
    true = 0
    total = 0
    while(count < len(x_train)/ batch_size - 1):
        names = x_train[100*count: 100*(count + 1)]
        labels = y_train[100*count: 100*(count + 1)]
        
        output, hidden = model(names)
        val, idx = hidden.max(1, keepdim=True)
        loss += criterion(hidden, torch.LongTensor(labels))
        count += 1
        
        for y_pred, y in zip(idx, labels):
            if y_pred == y:
                true+=1
            total += 1
        
    
    
    print("Epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print('Accuaracy: ',true/total)
    
    loss.backward()
    optimizer.step()    
true = 0
total = 0
count = 0
while(count < len(x_test)/ batch_size - 1):
    names = x_test[100*count: 100*(count + 1)]
    labels = y_test[100*count: 100*(count + 1)]

    output, hidden = model(names)
    val, idx = hidden.max(1, keepdim=True)
    count += 1

    for y_pred, y in zip(idx, labels):
        if y_pred == y:
            true+=1
        total += 1
print('Accuaracy: ',true/total)
