from random import randint



import torch

import torch.cuda as cuda

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import numpy as np



from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pack_sequence, pad_packed_sequence



device = "cuda" if cuda.is_available() else "cpu"

print(f"Using device {device}")
SEQ_LENGTH = 10

NO_SEQS = 1000



lengths = torch.randint(1, SEQ_LENGTH + 1, (NO_SEQS, ))



x = [ torch.randint(0, 10, (length, 1)).float().to(device) for length in lengths ]

padx = pad_sequence(x, batch_first=True)

pacx = pack_padded_sequence(padx, lengths, batch_first=True, enforce_sorted=False)

y = padx.sum(2).sum(1)



x_mean = torch.mean(pacx.data, dim=0)

x_std = torch.std(pacx.data, dim=0)

padx = (padx - x_mean) / x_std

pacx = pack_padded_sequence(padx, lengths, batch_first=True, enforce_sorted=False)



y_mean = y.mean()

y_std = y.std()

y = (y - y_mean) / y_std
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden_size = 4

        self.gru = nn.GRU(input_size=1, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        self.lin = nn.Linear(self.hidden_size, 1)



    def forward(self, x):

        _, o = self.gru(x)

        return self.lin( F.relu(o.view(-1, self.hidden_size), inplace=True) )
net = Net().to(device)



loss_fn = nn.MSELoss()

opt = optim.Adam(net.parameters(), lr=1e-3)



B = 16



for i in range(100):



    losses = []



    idx = torch.randperm(padx.shape[0])



    for b in range(0, padx.shape[0], B):



        bx = padx[idx[b:b+B]]

        by = y[idx[b:b+B]]

        bp = net(pack_padded_sequence(bx, lengths[idx[b:b+B]], batch_first=True, enforce_sorted=False))

        bp = bp.view(-1)



        loss = loss_fn(bp, by)

        net.zero_grad()

        loss.backward()

        opt.step()



        losses.append(loss.item())



    if (i+1) % 20 == 0:

        print("""

        Epoch    {}

        Loss     {}""".format(i+1, np.mean(losses)))
test_lengths = torch.randint(1, SEQ_LENGTH + 1, (10, ))



x_test = [ torch.randint(0, 10, (length, 1)).float().to(device) for length in test_lengths ]

padtx = pad_sequence(x_test, batch_first=True)

pactx = pack_padded_sequence(padtx, test_lengths, batch_first=True, enforce_sorted=False)

y_test = padtx.sum(2).sum(1)



padtx = (padtx - x_mean) / x_std

pactx = pack_padded_sequence(padtx, test_lengths, batch_first=True, enforce_sorted=False)



prediction = net(pactx).view(-1).detach()

print(prediction * y_std + y_mean)

print(y_test)