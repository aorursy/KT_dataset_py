with open('../input/alice-wonderland-dataset/alice_in_wonderland.txt', 'r', encoding='latin1') as f:
    data = f.read()
print("Extract: ", data[:50])
print("Length: ", len(data))
chars = list(set(data))
indexer = {char: index for (index, char) in enumerate(chars)}
indexed_data = []
for c in data:
    indexed_data.append(indexer[c])
    
print("Indexed extract: ", indexed_data[:50])
print("Length: ", len(indexed_data))
def index2onehot(batch):
    
    batch_flatten = batch.flatten()
    onehot_flat = np.zeros((batch.shape[0] * batch.shape[1], len(indexer)))
    onehot_flat[range(len(batch_flatten)), batch_flatten] = 1
    onehot = onehot_flat.reshape((batch.shape[0], batch.shape[1], -1))
    
    return onehot
import torch
from torch import nn
class LSTM(nn.Module):
    def __init__(self, char_length, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(char_length, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, char_length)
        
    def forward(self, x, states):
        out, states = self.lstm(x, states)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.output(out)
        
        return out, states
    
    def init_states(self, batch_size):
        hidden = next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size).zero_()
        cell = next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size).zero_()
        states = (hidden, cell)
        
        return states
import math
import numpy as np
n_seq = 100 ## Number of sequences per batch
seq_length =  50
n_batches = math.floor(len(indexed_data) / n_seq / seq_length)

total_length = n_seq * seq_length * n_batches
x = indexed_data[:total_length]
x = np.array(x).reshape((n_seq,-1))
model = LSTM(len(chars), 256, 2)
model
from torch import optim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20
import torch.nn.functional as F
losses = []
for e in range(1, epochs+1):
    states = model.init_states(n_seq)
    batch_loss = []
    
    for b in range(0, x.shape[1], seq_length):
        x_batch = x[:,b:b+seq_length]
        
        if b == x.shape[1] - seq_length:
            y_batch = x[:,b+1:b+seq_length]
            y_batch = np.hstack((y_batch, indexer["."] * np.ones((y_batch.shape[0],1))))
        else:
            y_batch = x[:,b+1:b+seq_length+1]
        
        x_onehot = torch.Tensor(index2onehot(x_batch))
        y = torch.Tensor(y_batch).view(n_seq * seq_length)
        
        pred, states = model(x_onehot, states)
        loss = loss_function(pred, y.long())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        batch_loss.append(loss.item())
        
    losses.append(np.mean(batch_loss))
    
    if e%1 == 0:
        print("epoch: ", e, "... Loss function: ", losses[-1])
import matplotlib.pyplot as plt
x_range = range(len(losses))
plt.plot(x_range, losses)
plt.xlabel("epochs")
plt.ylabel("Loss function")
plt.show()
starter = "So she was considering in her own mind "
states = None
for ch in starter:
    x = np.array([[indexer[ch]]])
    x = index2onehot(x)
    x = torch.Tensor(x)
    
    pred, states = model(x, states)

counter = 0
while starter[-1] != "." and counter < 50:
    counter += 1
    x = np.array([[indexer[starter[-1]]]])
    x = index2onehot(x)
    x = torch.Tensor(x)
    
    pred, states = model(x, states)
    pred = F.softmax(pred, dim=1)
    p, top = pred.topk(10)
    p = p.detach().numpy()[0]
    top = top.numpy()[0]
    index = np.random.choice(top, p=p/p.sum())
    
    starter += chars[index]
print(starter)