import numpy as np



import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
xy = np.loadtxt('../input/gif-for-gru-ver-2/train_data.csv',

                        delimiter=',', dtype=np.float32)

len = xy.shape[0]

x_data = torch.from_numpy(xy[:, 1:])

y_data = torch.from_numpy(xy[:, [0]]).reshape(2).long()

x_data.shape
x_data.view(2,5,16).shape
### make dataset iterable

class MyDataset(Dataset):



    def __init__(self):

        xy = np.loadtxt('../input/gif-for-gru-ver-2/train_data.csv',

                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:, 1:]).view(2,5,16)

        self.y_data = torch.from_numpy(xy[:, [0]]).reshape(2).long()



    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]



    def __len__(self):

        return self.len

    

dataset = MyDataset()

train_loader = DataLoader(dataset=dataset,

                          batch_size=1,

                          shuffle=False)
### create gru model

class MyGru(nn.Module):

    

    def __init__(self, input_size, hidden_size):

        super(MyGru, self).__init__()

        self.update_gate_for_input = nn.Linear(input_size, hidden_size)

        self.update_gate_for_hidden = nn.Linear(hidden_size, hidden_size)

        print(self.update_gate_for_input.weight)

        print(self.update_gate_for_hidden.weight)

        

        self.reset_gate_for_input = nn.Linear(input_size, hidden_size)

        self.reset_gate_for_hidden = nn.Linear(hidden_size, hidden_size)

        print(self.reset_gate_for_input.weight)

        print(self.reset_gate_for_hidden.weight)

        

        self.memory_gate_for_input = nn.Linear(input_size, hidden_size)

        self.memory_gate_for_hidden = nn.Linear(hidden_size, hidden_size)

        print(self.memory_gate_for_input.weight)

        print(self.memory_gate_for_hidden.weight)

        

        self.sig = nn.Sigmoid()

        self.tanh = nn.Tanh()        



     

    def forward(self, x, hidden):

        update_gate_for_input = self.update_gate_for_input(x)

        update_gate_for_hidden = self.update_gate_for_hidden(hidden).view(-1, hidden_size)

        update_gate_t = self.sig((update_gate_for_input + update_gate_for_hidden)) ### output after update gate

        

        reset_gate_for_input = self.reset_gate_for_input(x)

        reset_gate_for_hidden = self.reset_gate_for_hidden(hidden).view(-1, hidden_size)

        reset_gate_t = self.sig((reset_gate_for_input + reset_gate_for_hidden)) ### output after reset gate

        

        memory_gate_for_input = self.memory_gate_for_input(x)

        memory_gate_for_hidden = self.memory_gate_for_hidden(hidden).view(-1, hidden_size)

        memory_content = self.tanh((memory_gate_for_input + (reset_gate_t * memory_gate_for_hidden))) ### output for reset gate(affects how the reset gate do work)

        

        hidden = (update_gate_t * hidden) + ((1 - update_gate_t) * memory_content) # torch.ones(input_size, hidden_size)

        

        return hidden
class MyModel(nn.Module):

    

    def __init__(self, input_size, hidden_size, num_classes):

        super(MyModel, self).__init__()

        self.gru = MyGru(input_size, hidden_size)

        self.linear = nn.Linear(hidden_size, num_classes)

        

    def forward(self, x, hidden):

        hidden = self.gru(x, hidden)

        

        output = self.linear(hidden)

        output = output.view(-1, num_classes)



        return hidden, output
input_size = 16

hidden_size = 16

num_classes = 2



model = MyModel(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  
sequence_size = 5



hidden_zeros = torch.zeros(hidden_size)



for epoch in range(100):

    list_of_lost = []

    for i, data in enumerate(train_loader):

        loss = 0

        input, label = data

        label = label

        input = input.requires_grad_()

        optimizer.zero_grad()

        #print('label: ', label)

        #print('input: ', input)

        for j in range(sequence_size - 1):

            #print('input: ', input[0, j])

            if j == 0:

                hidden, output = model(input[0, j], hidden_zeros)

            else:

                hidden, output = model(input[0, j], hidden)

            #print(output)

            loss += criterion(output, label)

            list_of_lost.append(criterion(output, label).item())

        loss.backward()

        optimizer.step()

        

        

    print('epoch: ', epoch)

    print('max.loss: ', max(list_of_lost))