### for kaggle environment only

!cp ../input/snu-19-fall-itdl-hw4-dataset/* .
####### This Code should not be changed except 'USE_GPU'. Please mail to T/A if you must need to change with proper description.

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F 

import torch.optim as optim

import numpy as np

from torch.utils.data import Dataset, DataLoader

import time

import math



########### Change whether you would use GPU on this homework or not ############

USE_GPU = True

#################################################################################

if USE_GPU and torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')



vocab = open('vocab.txt').read().splitlines()

n_vocab = len(vocab)

torch.manual_seed(1)



# Change char to index.

def text2int(csv_file, dname, vocab):

    ret = []

    data = csv_file[dname].values

    for datum in data:

        for char in str(datum):

            idx = vocab.index(char)

            ret.append(idx)    

    ret = np.array(ret)

    return ret



# Create dataset to automatically iterate.

class NewsDataset(Dataset):

    def __init__(self, csv_file, vocab):

        self.csv_file = pd.read_csv(csv_file, sep='|')

        self.vocab = vocab

        self.len = len(self.csv_file)

        self.x_data = torch.tensor(text2int(self.csv_file, 'x_data', self.vocab))

        self.y_data = torch.tensor(text2int(self.csv_file, 'y_data', self.vocab))



    def __len__(self):

        return self.len



    def __getitem__(self, idx):

        return self.x_data[idx], self.y_data[idx]



dataset = NewsDataset(csv_file = 'data.csv', vocab = vocab)

train_loader = DataLoader(dataset=dataset, 

                             batch_size=64,

                             shuffle=False,

                             num_workers=1)
#################### WRITE DOWN YOUR CODE ################################

## Task_recommended form. You can use another form such as nn.Sequential or barebone Pytorch if you want, but in that case you may need to change some test or train code that given on later.

class selfModule(nn.Module):

    def __init__(self, input_size=n_vocab, hidden_size=128):

        super(selfModule, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.fc = nn.Sequential(

            nn.Linear(hidden_size, input_size),

            nn.LogSoftmax(dim=1)

        )



    def forward(self, x, hidden=None):

        one_hot_vector = self.one_hot_encode(x)

        x, hidden = self.lstm(one_hot_vector)

        x = self.fc(x.reshape(x.shape[0], -1))

        return x, hidden

    

    def one_hot_encode(self, x):

        one_hot_vector = torch.zeros((x.shape[0],1, self.input_size), dtype=torch.float).to(device=device)

        for idx, v in enumerate(x):

            one_hot_vector[idx][0][int(v)] = 1.

        return one_hot_vector

        

#################### WRITE DOWN YOUR CODE ################################
def int2text(x):

    return ''.join([vocab[i] for i in x])
def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)



###################### Train Code. On mostly you don't need to change this, but ok if you really need to do it.

def train(dataset, model, optimizer, n_iters): 

    model.to(device=device)

    model.train()

    start = time.time()

    print_every = 50

    for e in range(n_iters):

        for i, (x, y) in enumerate(dataset):

            x = x.to(device=device)

            y = y.to(device=device)

            model.zero_grad()

            output, _ = model(x)

            loss = loss_fcn(output, y)

            loss.backward(retain_graph=True)

            optimizer.step()

        if e % print_every == 0:

            print('%s (%d %d%%) %.4f' % (timeSince(start), e, e / n_iters * 100, loss))
####################### Test Code. On mostly you don't need to change this except value of 'max_length', but ok if you really need to do it.

def test(start_letter):

    max_length = 1000    

    with torch.no_grad():

        idx = vocab.index(start_letter)

        input_nparray = [idx]

        input_nparray = np.reshape(input_nparray, (1, len(input_nparray)))

        inputs = torch.tensor(input_nparray, device=device, dtype=torch.long)

        output_sen = start_letter

        model.eval()

        for i in range(max_length):

            output, _ = model(inputs)

            topv, topi = output.topk(1)

            topi = topi[-1]

            letter = vocab[topi]

            output_sen += letter

            idx = vocab.index(letter)

            input_nparray = np.append(input_nparray, [idx])

            inputs = torch.tensor(input_nparray, device=device, dtype=torch.long)

    return output_sen
print('using device:', device)

n_hidden = 128

loss_fcn = nn.NLLLoss()

model = selfModule(n_vocab, n_hidden)



do_restore = False



if do_restore:

    model.load_state_dict(torch.load('fng_pt.pt'))

    model.eval()

    model.to(device=device)

else:

    n_iters = 500

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=2e-16, weight_decay=0)

    train(train_loader, model, optimizer, n_iters)

    torch.save(model.state_dict(), 'fng_pt.pt')



print(test('W'))