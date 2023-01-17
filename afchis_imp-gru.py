import numpy as np



import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):



    def __init__(self):

        xy = np.loadtxt('../input/gif-for-gru/train_data.csv',

                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:, 1:])

        self.y_data = torch.from_numpy(xy[:, [0]]).reshape(10).long()



    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]



    def __len__(self):

        return self.len

    

dataset = MyDataset()

train_loader = DataLoader(dataset=dataset,

                          batch_size=1,

                          shuffle=False)
input_size = 16

hidden_size = 32

sequence_length = 1

num_classes = 2

batch_size = 1

num_layers = 1
class Model(nn.Module):



    def __init__(self):

        super(Model, self).__init__()

        self.rnn = nn.GRU(input_size,

                          hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)



    def forward(self, hidden, x):

        # Reshape input (batch first)

        x = x.view(batch_size, sequence_length, input_size) 



        # Propagate input through RNN

        # Input: (batch, seq_len, input_size)

        # hidden: (num_layers * num_directions, batch, hidden_size)

        out, hidden = self.rnn(x, hidden)

        out = self.fc(out[:, -1, :])

        #print(out)

        return hidden, out
model = Model()

print(model)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
h_0 = torch.zeros(num_layers, batch_size, hidden_size).requires_grad_()

for epoch in range(20):

    l = []

    for i, data in enumerate(train_loader):

        #print(i)

        inputs, labels = data

        inputs, labels = inputs.requires_grad_(), labels

        optimizer.zero_grad()

    

        if i < len(train_loader)/2:

            #print(i)

            if i == 0:

                hidden, outputs = model(h_0, inputs)

            elif i == (len(train_loader)/2 - 1):

                hidden, outputs = model(hidden, inputs)

                loss = criterion(outputs, labels)

                l.append(loss.item())

                loss.backward()

                optimizer.step()

            else:

                hidden, outputs = model(hidden, inputs)

        else :

            #print(i)

            if i == len(train_loader)/2:

                hidden, outputs = model(h_0, inputs)

            elif i == (len(train_loader) - 1):

                hidden, outputs = model(hidden, inputs)

                loss = criterion(outputs, labels)

                l.append(loss.item())

                loss.backward()

                optimizer.step()

            else:

                hidden, outputs = model(hidden, inputs)

            

        #print(loss.item())

    print('epoch: ', epoch)

    print('max.loss: ', max(l))

        # Run your training process

        #print(epoch, i, "inputs", inputs.data, "labels", labels.data)