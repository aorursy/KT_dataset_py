# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
def _normalize(feature, train_size):

    #

    feature_t = feature[:train_size]

    mean = feature_t.mean(axis=0)

    std = feature_t.std(axis=0)

    feature_n = (feature - mean) / std

    return feature_n
class AverageMeter(object):

    """Computes and stores the average and current value"""



    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
def load_csv(file_path="/kaggle/input/boston-house-prices", file_name='housing.csv'):

    input = []

    target = []

    with open(os.path.join(file_path, file_name)) as f:

        for line in f.readlines():

            data_list = line.strip().split()

            input.append(data_list[:-1])

            target.append(data_list[-1])

    input = np.array(input).astype(float)

    target = np.array(target).astype(float)

    print(input.shape, target.shape)

    return input, target



def construct_data_iter(input, target):

    input = torch.from_numpy(input).float()

    target = torch.from_numpy(target).float()

    print("constructing data iterator", input.shape, target.shape)

    deal_dataset = TensorDataset(input, target)

    data_loader = DataLoader(dataset=deal_dataset, batch_size=32, shuffle=True, num_workers=2)

    return data_loader
class RegressionModel(nn.Module):

    def __init__(self, input_size: int, dims: list, dropout: float = 0.5):

        super(RegressionModel, self).__init__()

        layers = []

        layers.append(nn.Linear(input_size, dims[0]))

        layers.append(nn.ReLU())

        for i in range(len(dims)-2):

            layers.append(nn.Linear(dims[i], dims[i+1]))

            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))

        layers.append(nn.Dropout(dropout))

        layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], 1))



        self.model = nn.Sequential(*layers)



    def forward(self, x):

        # x = torch.tensor(x, dtype=torch.float32)

        return self.model(x)

Model = RegressionModel(13, [100, 50, 20])

print(Model)
# choose SmoothL1Loss() as our loss function/ criterion.

criterion = torch.nn.SmoothL1Loss()

# show our dataset size.

train_size, valid_size, test_size = 400, 56, 50

def main():

    x, y = load_csv()

    x = _normalize(x, train_size)

    train_data_iter = construct_data_iter(x[:train_size], y[:train_size])

    valid_data_iter = construct_data_iter(x[train_size:train_size+valid_size], y[train_size:train_size+valid_size])

    Model = RegressionModel(13, [256, 256, 128, 128, 128])

    ADAM_optimizer = optim.Adam(Model.parameters(), lr=0.0001)

    SGD_optimizer = optim.SGD(Model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001)



    def forward(data, optimizer=None, Training=True):

        perplexity = AverageMeter()

        if Training:

            Model.train()

        else:

            Model.eval()

        for i, (Input, Target) in enumerate(data):

            pred = Model(Input)

            err = criterion(pred.squeeze(), Target)

            perplexity.update(float(err.item()))

            if Training:

                optimizer.zero_grad()

                err.backward()

                optimizer.step()

            if i % 15 == 14:

                print('{phase} - Epoch: [{0}][{1}/{2}]\t' 'Perplexity {perp.val:.4f} ({perp.avg:.4f})'.format(

                    epoch, i, len(data),

                    phase='TRAINING' if Training else 'EVALUATING',

                    perp=perplexity))

        return perplexity.avg



    for epoch in range(500):

        train_prep = forward(train_data_iter, SGD_optimizer, Training=True)

        # evaluate

        val_prep = forward(valid_data_iter, Training=False)

        if epoch % 25 == 0:

            print('Epoch: {0}\tTraining Perplexity {1} \tValidation Perplexity {val_perp:.4f} \n'.format(epoch + 1, train_prep, val_perp=val_prep))



    print("testing our model ...")



    for i in range(506-test_size, 506):

        Model.eval()

        pred = Model.forward(torch.from_numpy(x[i]).float()).squeeze()

        ground_true = y[i]

        print("test result : ", i, pred.item(), ground_true)
main()