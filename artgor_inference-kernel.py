# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from typing import List, Dict, Optional



import numpy as np

from torch.utils.data import Dataset



import pandas as pd

import torch

from sklearn.model_selection import train_test_split

import torch

from torch import nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import math

from torch.utils.data import TensorDataset, DataLoader

from typing import Dict, Union

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

s = pd.DataFrame({'sig_id': test_features['sig_id'].values})
n_h_layers = 2048

learning_rate = 5e-3

criterion = nn.BCEWithLogitsLoss()



class Net(nn.Module):

    def __init__(self, n_in, n_h, n_out, n_out1):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_in, n_h)

        self.fc2 = nn.Linear(n_h, math.ceil(n_h/4))

        self.fc3 = nn.Linear(math.ceil(n_h/4), n_out)

        self.fc4 = nn.Linear(math.ceil(n_h/4), n_out1)

        self.bn = nn.BatchNorm1d(n_in)

        self.bn1 = nn.BatchNorm1d(n_h)

        self.bn2 = nn.BatchNorm1d(math.ceil(n_h/4))

        self.drop = nn.Dropout(0.4)

        self.n_out = n_out

        self.selu = nn.SELU()

        self.sigm = nn.Sigmoid()

    def forward(self, x, targets, targets1):

        

        

        self.loss = criterion

        x = self.fc1(self.bn(x))

        x = self.selu(x)

        x = self.fc2(self.drop(self.bn1(x)))

        x = self.selu(x)

        

        # scored targets

        x1 = self.fc3(self.bn2(x))

        # non scored targets

        x2 = self.fc4(self.bn2(x))

        loss = (self.loss(x1, targets) + self.loss(x2, targets1)) / 2

        real_loss = self.loss(x1, targets)

        # probabilities

        out = self.sigm(x1)

        return out, loss, real_loss

    

net = Net(n_in = 879, n_h = n_h_layers, n_out = 206, n_out1 = 402)
class MoADataset(Dataset):

    def __init__(

        self,

        data,

        targets = None,

        targets1 = None,

        mode = 'train'

    ):

        """



        Args:

        """



        self.mode = mode

        self.data = data

        self.targets = targets

        self.targets1 = targets1



    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        data = self.data[idx]

        if self.targets is not None:

            target = self.targets[idx]

            target1 = self.targets1[idx]

        else:

            target = np.zeros((206,))

            target1 = np.zeros((402,))

            

        sample = {'data': torch.tensor(data).float(),

                  'target': torch.tensor(target).float(),

                  'target1': torch.tensor(target1).float()}



        return sample



    def __len__(self) -> int:

        return len(self.data)
test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_time'], prefix='cp_time')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_dose'], prefix='cp_dose')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_type'], prefix='cp_type')], axis=1)

# test_features = test_features.loc[test_features['cp_type'] != 'ctl_vehicle']

test_features = test_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
test_dataset = MoADataset(data=test_features.iloc[:, 1:].values)

test_loader = torch.utils.data.DataLoader(

            test_dataset,

            batch_size=1024,

            num_workers=0,

            shuffle=False,

        )
net.load_state_dict(torch.load('/kaggle/input/lish-moa-baseline-approach/model.pt'))

net.eval()
predictions = np.zeros((test_features.shape[0], 206))

for ind, batch in enumerate(test_loader):

    p = net(batch['data'], batch['target'], batch['target1'])[0].detach().cpu().numpy()

    predictions[ind * 1024:(ind + 1) * 1024] = p
for col in train_targets_scored.columns[1:].tolist():

    s[col] = 0
s.loc[s['sig_id'].isin(test_features['sig_id']), train_targets_scored.columns[1:]] = predictions
s.to_csv('submission.csv', index=False)
plt.hist(predictions.mean())

plt.title('Distribution of prediction means');