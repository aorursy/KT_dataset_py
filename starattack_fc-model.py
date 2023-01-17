import torch

import torch.nn as nn

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils import data





learning_rate = 0.001

batch_size = 50

epochs = 40





network = nn.Sequential(

            nn.Linear(784, 128),

            nn.Linear(128, 64),

            nn.Linear(64, 10),

            nn.Softmax()

)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)





train_data_file = '/kaggle/input/digit-recognizer/train.csv'

train_data = pd.read_csv(train_data_file).to_numpy()

data_input = train_data[:, 1:] / 255.0

data_label = train_data[:, :1]

train_input, val_input, train_label, val_label = train_test_split(data_input, data_label, test_size=0.2, random_state=42)

train_input = torch.Tensor(train_input)

train_label = torch.from_numpy(train_label).type(torch.LongTensor).squeeze()

val_input = torch.Tensor(val_input)

val_label = torch.from_numpy(val_label).type(torch.LongTensor).squeeze()

train_data = data.TensorDataset(train_input, train_label)

val_data = data.TensorDataset(val_input, val_label)

train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)





for epoch in range(epochs):

    print("epoch:%d" % epoch)

    for i, (batch_input, batch_label) in enumerate(train_loader):

        optimizer.zero_grad()

        output = network(batch_input)

        loss = loss_function(output, batch_label)

        loss.backward()

        optimizer.step()

    total = 0.0

    correct = 0.0

    for test_input, test_label in val_loader:

        total += len(test_label)

        output = network(test_input)

        output = torch.max(output.data, 1)[1]

        correct += (output == test_label).sum()

    accuracy = correct / total

    print("total:%d, correct:%d, accuracy:%f" % (total, correct, accuracy))

        



test_data_file = '/kaggle/input/digit-recognizer/test.csv'

test_data = pd.read_csv(test_data_file).to_numpy()

test_data = torch.Tensor(test_data)

print(test_data.shape)

result = []

for index in range(test_data.shape[0]):

    test_input = test_data[index : index+1, :]

    test_output = network(test_input)

    digit = np.argmax(test_output[0].detach().numpy())

    result.append([index + 1, digit])

result = np.array(result)

print(result.shape)

data_frame = pd.DataFrame(result, columns=['ImageId', 'Label'])

data_frame.to_csv('result.csv',  index=False)