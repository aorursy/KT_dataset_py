import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import datetime



import numpy as np

import random



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.data as tud

import torch.optim as optim



# set cuda

USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')



#set random seed

RANDOM_SEED = 10015

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if USE_CUDA:

    torch.cuda.manual_seed(RANDOM_SEED)

    

# set hyper parameters



# LSTM

NUM_HIDDEN = 256

N_STEP = 66



# data

TIME_STEP = 67

NUM_FEATURES = 2



# train

BATCH_SIZE = TIME_STEP // N_STEP

EPOCHS = 2000

LEARNING_RATE = 0.01

CLIP_VALUE = 1.



# predict

N_PREFIX = 10

N_PREDICT = 33
class COVID19Dataset(tud.Dataset):

    def __init__(self, dataframe, time_step, n_step):

        super(COVID19Dataset, self).__init__()

        self.data = torch.tensor(

            dataframe.iloc[:, -2:].values, dtype=torch.float32, device=DEVICE)

        self.data_length = len(self.data)

        self.time_step = time_step

        self.n_step = n_step

        self.num_features = self.data.shape[-1]

        assert self.data_length % self.time_step == 0

        self.num_region = self.data_length // self.time_step

        self.steps_per_region = self.time_step // self.n_step



        # reshape data: (num_region, time_step, num_features)

        self.data = self.data.view(-1, self.time_step, self.num_features)

        assert self.data.shape == torch.Size(

            [self.num_region, self.time_step, self.num_features])



    def __len__(self):

        return self.steps_per_region * self.num_region



    def __getitem__(self, idx):

        idx_region = idx // self.steps_per_region

        idx_n_step = idx % self.steps_per_region

        data_features = self.data[idx_region, idx_n_step *

                                  self.n_step: (idx_n_step+1) * self.n_step, :]

        data_labels = self.data[idx_region, idx_n_step *

                                self.n_step + 1: (idx_n_step+1) * self.n_step + 1, :]

        

        return data_features, data_labels
train_df = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')



train_ds = COVID19Dataset(train_df, TIME_STEP, N_STEP)

# data_features: (B, n_step, num_features)

# data_labels: (B, n_step, num_features)

train_dl = tud.DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
class COVID19Model(nn.Module):

    def __init__(self, num_features, n_hidden, n_layers=1):

        super(COVID19Model, self).__init__()

        self.num_features = num_features

        self.n_hidden = n_hidden

        self.n_layers = n_layers



        self.lstm = nn.LSTM(num_features, n_hidden, n_layers)

        self.dense1 = nn.Linear(n_hidden, 128)

        self.bnorm1 = nn.BatchNorm1d(128)

        self.dense2 = nn.Linear(128, 64)

        self.bnorm2 = nn.BatchNorm1d(64)

        self.dense3 = nn.Linear(64, 32)

        self.bnorm3 = nn.BatchNorm1d(32)

        self.dense4_1 = nn.Linear(32, 1)

        self.dense4_2 = nn.Linear(32, 1)

        self.drop = nn.Dropout(0.3)



        self.init_weights()



    def forward(self, x, hidden):

        # reshape input x:

        x = x.permute(1, 0, 2).contiguous()  # (n_step, B, num_features)

        x, hidden = self.lstm(x, hidden)  # output0: (n_step, B, n_hidden)

        x = self.drop(x)

        x = x.view(-1, x.shape[-1])  # x: (n_step * B, n_hidden)

        x = F.relu(self.bnorm1(self.dense1(x)))

        x = self.drop(x)

        x = F.relu(self.bnorm2(self.dense2(x)))

        x = self.drop(x)

        x = F.relu(self.bnorm3(self.dense3(x)))

        x = self.drop(x)

        output1 = torch.clamp(self.dense4_1(x), 0)  # output1: (n_step * B, 1)

        output2 = torch.clamp(self.dense4_2(x), 0)  # output2: (n_step * B, 1)



        return (output1, output2), hidden

    

    def init_weights(self):

        initrange = 0.1

        self.dense1.weight.data.uniform_(-initrange, initrange)

        self.dense2.weight.data.uniform_(-initrange, initrange)

        self.dense3.weight.data.uniform_(-initrange, initrange)

        self.dense4_1.weight.data.uniform_(-initrange, initrange)

        self.dense4_2.weight.data.uniform_(-initrange, initrange)

        self.dense1.bias.data.zero_()

        self.dense2.bias.data.zero_()

        self.dense3.bias.data.zero_()

        self.dense4_1.bias.data.zero_()

        self.dense4_2.bias.data.zero_()

        

    def init_hidden(self, batch_size, requires_grad=True):

        weight = next(self.parameters())

        return (weight.new_zeros(self.n_layers, batch_size, self.n_hidden, requires_grad=requires_grad),

                weight.new_zeros(self.n_layers, batch_size, self.n_hidden, requires_grad=requires_grad))
model = COVID19Model(NUM_FEATURES, NUM_HIDDEN)

if USE_CUDA:

    model.cuda()
loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.93)
def get_loss(outputs, labels):

    output1, output2 = outputs

    label1 = labels[:, 0].unsqueeze(-1)

    label2 = labels[:, 1].unsqueeze(-1)

    

    loss1 = loss_fn(output1, label1)

    loss2 = loss_fn(output2, label2)

    

    return loss1 + loss2
total_epoch_loss = []

min_loss = None

for epoch in range(EPOCHS):

    epoch_loss = 0

    model.train()

    for i, batch in enumerate(train_dl):

        data_features, data_labels = batch

        # reshape data_labels: (B, n_step, num_features) -> (B * n_step, num_features)

        data_labels = data_labels.permute(

            1, 0, 2).contiguous().view(-1, data_labels.shape[-1])



        if USE_CUDA:

            data_features = data_features.cuda()

            data_labels = data_labels.cuda()



        hidden = model.init_hidden(BATCH_SIZE)

        hidden = tuple(c.detach() for c in hidden)

            

        outputs, hidden = model(data_features, hidden)  # outputs: (output1, output2)

        loss = get_loss(outputs, data_labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)

        optimizer.step()

        model.zero_grad()

        

        epoch_loss += loss.item()

    total_epoch_loss.append(epoch_loss / i)

    print('epoch{}, loss: '.format(epoch + 1), total_epoch_loss[-1])

    if epoch % 20 == 0:

        if min_loss == None or total_epoch_loss[-1] < min_loss:

            min_loss = total_epoch_loss[-1]

            torch.save(model.state_dict(), 'covid19-best.th')

        else:

            scheduler.step()

            print('learning rate changed! learning rate: ', scheduler.get_lr())
plt.plot(total_epoch_loss)
test_data = torch.tensor(train_df.iloc[:, -2:].values, dtype=torch.float32)

test_data = test_data.view(-1, TIME_STEP, 2)

test_data = test_data[:, (TIME_STEP-N_PREFIX):, :]

if USE_CUDA:

    test_data = test_data.cuda()
# model.load_state_dict(torch.load('../input/roger10015-rnn-covid19/covid19-best.th'))

model.eval()

num_region = len(train_df) // TIME_STEP

predict_output = []

with torch.no_grad():

    for region in range(num_region):

        region_predict = []

        region_predict += [test_data[region, 0, :]]



        hidden = model.init_hidden(1, requires_grad=False)

        for step in range(N_PREFIX + N_PREDICT - 1):

            (outputs1, outputs2), hidden = model(

                region_predict[-1].view(1, 1, -1), hidden)

            outputs1 = outputs1.round()

            outputs2 = outputs2.round()

            outputs = torch.cat((outputs1, outputs2)).squeeze(-1)

            if step < N_PREFIX - 1:

                region_predict += [test_data[region, step + 1, :]]

            else:

                region_predict += [outputs]

        predict_output += region_predict

predict_output = [list(t.long().cpu().numpy()) for t in predict_output]
predict_output_df = pd.DataFrame(predict_output)
submission = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')

submission['ConfirmedCases'] = predict_output_df.iloc[:, 0]

submission['Fatalities'] = predict_output_df.iloc[:, 1]
submission.to_csv('submission.csv', index=False)