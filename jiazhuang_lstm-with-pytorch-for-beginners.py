import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

%matplotlib inline

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(4)
train = pd.read_csv('../input/train.csv', dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
train.head()
sample_freq = 100

plt.plot(train.acoustic_data.values[::sample_freq])

plt.plot(train.time_to_failure.values[::sample_freq]*100)
def extract_features(z):

    return np.c_[

        z.mean(axis=1), 

        np.percentile(np.abs(z), q=[0, 25, 50, 75, 100], axis=1).T, 

        z.std(axis=1)

    ]
def create_X(x, window_size=1000, sequence_len=150):

    tmp = x.reshape(sequence_len, -1)

    return np.c_[

        extract_features(tmp),

        extract_features(tmp[:, -window_size // 10:]),

        extract_features(tmp[:, -window_size // 100:])

    ]
n_features = create_X(train.acoustic_data.values[0:150000]).shape[1]
class TrainData(Dataset):

    def __init__(self, df, window_size=1000, sequence_len=150):

        self.rows = df.shape[0] // (window_size*sequence_len)

        self.data, self.labels = [], []

        for segment in range(self.rows):

            seg = df.iloc[segment*window_size*sequence_len: (segment+1)*window_size*sequence_len]

            x = seg.acoustic_data.values

            y = seg.time_to_failure.values[-1]

            self.data.append(create_X(x))

            self.labels.append(y)

    

    def __len__(self):

        return self.rows

    

    def __getitem__(self, idx):

        return (

            torch.from_numpy(self.data[idx].astype(np.float32)),

            self.labels[idx]

        )
train_data = TrainData(train)
batch_size = 100

n_steps = len(train_data) // 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    

    def forward(self, x):

        hidden = (

            torch.zeros(1, x.size(0), self.hidden_size),

            torch.zeros(1, x.size(0), self.hidden_size)

        )

        

        out, _ = self.lstm(x, hidden)

        

        out = self.fc(out[:, -1, :])

        return out.view(-1)
input_size = n_features

hidden_size = 32

model = LSTM(input_size, hidden_size)
learning_rate = 0.01

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(2):

    for i, (data, labels) in enumerate(train_loader):

        outputs = model(data)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i % 10 == 0:

            print(f'[Epoch {epoch}/2, Step {i}/{n_steps}]  loss: {loss.item(): .4f}')