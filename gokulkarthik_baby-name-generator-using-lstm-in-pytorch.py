import os

import time

from tqdm.notebook import tqdm



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

#from torchsummary import summary



import string

from collections import Counter
#writer = SummaryWriter(os.path.join("runs", "baby-names"))
data_path = os.path.join("/kaggle", "input", "us-baby-names", "NationalNames.csv")

data = pd.read_csv(data_path)

print(data.shape)

data.head()
data.info()
def clean(name):

    

    name = name.lower().strip()

    name = "".join([c for c in name if c in string.ascii_lowercase])

    name += "."

    return name
data['Name'] = data['Name'].apply(clean)

data.head()
names = data[['Name', 'Count']].groupby('Name').sum()

del names.index.name

print(len(names))

names.head()
pd.Series(names.index).apply(len).max()
max_length = 11

len_filter = pd.Series(names.index).apply(lambda x: len(x)<=max_length).tolist() # max length of 10 excluding '.'

print(len_filter[:10])

print(names.shape)

names = names[len_filter]

print(names.shape)
pd.Series(names.index).apply(len).max()
names = names.sort_values(by=['Count'], ascending=False)

names.head()
names['Count'].describe()
alpha = 0.8

names['Count'].apply(lambda x: np.power(x, alpha)).apply(np.int).describe()
names['count_normalized'] = names['Count'].apply(lambda x: np.power(x, alpha)).apply(np.int)

names.head()
count_normalized_sum = names['count_normalized'].sum()

print(count_normalized_sum)
names['p'] = names['count_normalized'] / count_normalized_sum

names.head()
np.random.seed(0)

names_list = np.random.choice(names.index, size=10**5, p=names['p'], replace=True)

print(len(names_list))

print(names_list[:50])
pd.Series(names_list).value_counts()
del data, names
chars = "." + string.ascii_lowercase

num_chars = len(chars)

print(chars)

print(num_chars)
char_to_id = {c:i for i, c in enumerate(chars)}

id_to_char = {v:k for k, v in char_to_id.items()}

print(char_to_id)

print(id_to_char)
print(max_length)
class NamesDataset(Dataset):

    

    def __init__(self, names_list):

        self.names_list = names_list

        

    def __len__(self):

        return len(self.names_list)

    

    def __getitem__(self, idx):

        x_str = self.names_list[idx].ljust(max_length, ".")[:max_length]

        y_str = x_str[1:] + "."

        

        x = torch.zeros((max_length, num_chars))

        y = torch.zeros(max_length)

        for i, c in enumerate(x_str):

            x[i, char_to_id[c]] = 1

        for i, c in enumerate(y_str):

            y[i] = char_to_id[c]

            

        return x, y
trainset = NamesDataset(names_list)
train_batch_size = 256
cpu_count = os.cpu_count()

print(cpu_count)
train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=cpu_count)

print(len(train_loader))
train_iter = iter(train_loader)

X, Y = train_iter.next()

print(X.size(), Y.size())
input_size = num_chars

hidden_size = 54

output_size = num_chars

num_layers = 1
device = "cuda:0" if torch.cuda.is_available() else "cpu"

device = torch.device(device)
class Model(nn.Module):

    

    def __init__(self, input_size, hidden_size, output_size, num_layers):

        super(Model, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc2 = nn.Linear(hidden_size, output_size)

        self.fc3 = nn.Linear(output_size, output_size)

        

    def forward(self, X, states):

        ht, ct = states

        batch_size = X.size(0)

        out, (ht, ct) = self.lstm1(X, (ht, ct))

        out = F.relu(self.fc2(out))

        out = self.fc3(out)

        return out, (ht, ct) # out: Size([batch_size, max_length, num_chars])
model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

model = nn.DataParallel(model)

model = model.to(device)
#list(model.parameters())
ht = torch.zeros((num_layers, train_batch_size, hidden_size)).to(device)

ct = torch.zeros((num_layers, train_batch_size, hidden_size)).to(device)

#writer.add_graph(model, (X, (ht, ct)))

#writer.close()
#summary(model, input_size=(max_length, num_chars))
lr = 0.005

step_size = len(train_loader) * 1

gamma = 0.95

print(step_size)
criterion = nn.CrossEntropyLoss(reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=lr)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
def generate_name(model, start='a', k=5):

    

    if len(start) >= max_length:

        return name

    

    with torch.no_grad():

        

        ht = torch.zeros((num_layers, 1, hidden_size)).to(device)

        ct = torch.zeros((num_layers, 1, hidden_size)).to(device)

        length = 0

        name = start

        

        for char in start:

            X = torch.zeros((1, 1, num_chars)) # [batch_size, timestep, num_chars]

            X[0, 0, char_to_id[char]] = 1

            out, (ht, ct) = model(X, (ht, ct))

            length += 1

        vals, idxs = torch.topk(out[0], k) # 0 -> first eg in a batch

        idx = np.random.choice(idxs.cpu().numpy()[0]) # 0 -> first...

        char = id_to_char[idx]

        vals, idxs = torch.topk(out[0], k) # 0 -> first eg in a batch

        idx = np.random.choice(idxs.cpu().numpy()[0]) # 0 -> first...

        char = id_to_char[idx]

        

        while char != "." and length <= max_length-1:

            X = torch.zeros((1, 1, num_chars)) # [batch_size, timestep, num_chars]

            X[0, 0, char_to_id[char]] = 1

            out, (ht, ct) = model(X, (ht, ct))

            vals, idxs = torch.topk(out[0], k) # 0 -> first eg in a batch

            idx = np.random.choice(idxs.cpu().numpy()[0]) # 0 -> first...

            char = id_to_char[idx]

            length += 1

            name += char

    

        if name[-1] != ".":

            name += "."

    

    return name
def sampler(model, start='a', n=10, k=5, only_new=False):

    

    names = []

    cnt = 0

    while cnt <= n:

        name = generate_name(model=model, start=start, k=k)

        if only_new: 

            if name not in names_list and name not in names:

                names.append(name)

                cnt += 1

        else:

            if name not in names:

                names.append(name)

                cnt += 1

    names = [name[:-1].title() for name in names]

    

    return names
epochs = 50

print_every_n_epochs = epochs // 10
epoch_losses = []

epoch_lrs = []

iteration_losses = []

iteration_lrs = []



for epoch in tqdm(range(1, epochs+1), desc="Epochs"):

    epoch_loss = 0

    epoch_lr = 0

    

    for i, (X, Y) in tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="Epoch-{}".format(epoch)):

    #for i, (X, Y) in enumerate(train_loader, 1):

        X, Y = X.to(device), Y.to(device)

        

        ht = torch.zeros((num_layers, X.size(0), hidden_size)).to(device)

        ct = torch.zeros((num_layers, X.size(0), hidden_size)).to(device)



        optimizer.zero_grad()

        Y_pred_logits, (ht, ct) = model(X, (ht, ct))

        Y_pred_logits = Y_pred_logits.transpose(1, 2) # Check Loss Doc: [N, d1, C] -> [N, C, d1]

        loss = criterion(Y_pred_logits, Y.long())

        loss.backward(retain_graph=True)

        optimizer.step()

        lr_scheduler.step()

        

        iteration_losses.append(loss.item())

        iteration_lrs.append(lr_scheduler.get_lr()[0])

        epoch_loss += loss.item()

        epoch_lr += lr_scheduler.get_lr()[0]

        

    epoch_loss /= len(train_loader)

    epoch_lr /= len(train_loader)

    epoch_losses.append(epoch_loss)

    epoch_lrs.append(epoch_lr)

    

    if epoch % print_every_n_epochs == 0:    

        message = "Epoch:{}    Loss:{}    LR:{}".format(epoch, epoch_loss, epoch_lr)

        print(message)

        names = sampler(model, start='jo', n=10, k=10, only_new=False)

        print(names)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

ax1.plot(epoch_losses, marker="o", markersize=5)

ax1.set_title("Loss")

ax2.plot(epoch_lrs, marker="o", markersize=5)

ax2.set_title("LR")

plt.xlabel("Epochs")

plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

ax1.plot(iteration_losses[::])

ax1.set_title("Loss")

ax2.plot(iteration_lrs[::])

ax2.set_title("LR")

plt.xlabel("Iterations")

plt.show()
window = 100

plt.figure(figsize=(15, 4))

pd.Series(iteration_losses).rolling(window=window).mean().iloc[window-1:].plot()

plt.show()
path = os.path.join("/kaggle", "working", "classifier.pth")

torch.save(model.state_dict(), path)
path = os.path.join("/kaggle", "working", "classifier.pth")

model = Model(input_size=num_chars, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

model = nn.DataParallel(model)

model.load_state_dict(torch.load(path))
names = sampler(model, start='indi', n=10, k=5, only_new=True)

print(names)
names = sampler(model, start='herb', n=10, k=5, only_new=False)

print(names)
names = sampler(model, start='su', n=10, k=5, only_new=True)

print(names)
names = sampler(model, start='vis', n=10, k=5, only_new=True)

print(names)
names = sampler(model, start='a', n=10, k=3, only_new=True)

print(names)
names = sampler(model, start='a', n=10, k=8, only_new=True)

print(names)
names = sampler(model, start='a', n=10, k=15, only_new=True)

print(names)
names = sampler(model, start='jam', n=10, k=2, only_new=False)

print(names)