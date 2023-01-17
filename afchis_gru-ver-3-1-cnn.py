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

        self.x_data = torch.from_numpy(xy[:, 1:]).view(2,5,4,4)

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

    

    def __init__(self):

        super(MyGru, self).__init__()

        self.update_gate = nn.Conv2d(in_channels=1+hidden_channel, out_channels=hidden_channel, kernel_size=3, stride=1, padding=1)

        

        self.reset_gate = nn.Conv2d(in_channels=1+hidden_channel, out_channels=hidden_channel, kernel_size=3, stride=1, padding=1)

        

        self.memory_gate_for_input = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)

        self.memory_gate_for_hidden = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)

        

        self.sig = nn.Sigmoid()

        self.tanh = nn.Tanh()        



     

    def forward(self, x, hidden):

        x = x.view(1, 1, 4, 4)

        

        input = torch.cat([x, hidden],dim=1)



        update_gate = self.update_gate(input)

        update_gate = self.sig((update_gate)) ### output after update gate

        

        reset_gate = self.reset_gate(input)

        reset_gate = self.sig((reset_gate)) ### output after reset gate

        

        memory_gate_for_input = self.memory_gate_for_input(x)

        memory_gate_for_hidden = self.memory_gate_for_hidden(hidden)



        memory_content = self.tanh((memory_gate_for_input + (reset_gate * memory_gate_for_hidden))) ### output for reset gate(affects how the reset gate do work)

        

        hidden = (update_gate * hidden) + ((1 - update_gate) * memory_content) # torch.ones(input_size, hidden_size)

        

        return hidden
class MyModel(nn.Module):

    

    def __init__(self, input_size, hidden_channel, num_classes):

        super(MyModel, self).__init__()

        

        self.gru = MyGru()

        self.linear = nn.Linear(input_size, num_classes)

        

    def forward(self, x, hidden):

        #GRU

        hidden = self.gru(x, hidden)

        #hidden_for_out = hidden.view(1, hidden.shape[1], -1).max(dim=-1)[0]

        hidden_for_out = hidden.max(dim=1)[0]

        hidden_for_out = hidden_for_out.view(-1)



        #Linear

        output = self.linear(hidden_for_out)

        output = output.view(-1, num_classes)

        

        return hidden, output
m, n = 4, 4

x = torch.randn([1,4,m,n])

print(x.shape)

print(x)

#x=x.view(1,4,-1)

#print(x.shape)

out=x.max(dim=1)[0]

print(out)
input_size = 4*4

hidden_channel = 4

num_classes = 2





model = MyModel(input_size, hidden_channel, num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)  
sequence_size = 5



hidden_zeros = torch.zeros(1, 4, 4, 4).to(device)



for epoch in range(1000):

    list_of_lost = []

    for i, data in enumerate(train_loader):

        loss = 0

        input, label = data

        label = label.to(device)

        input = input.to(device)

        optimizer.zero_grad()

        for j in range(sequence_size):

            if j == 0:

                hidden, output = model(input[0, j], hidden_zeros)

            else:

                hidden, output = model(input[0, j], hidden)

            

            loss += criterion(output, label)

            list_of_lost.append(criterion(output, label).item())

        loss.backward()

        optimizer.step()

        

        

    if epoch % 10 == 0:

        print('epoch: ', epoch)

        print('max.loss: ', max(list_of_lost))

    