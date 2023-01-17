# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



import datetime

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.autograd import Variable
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
sns.distplot(df_train['299'])
from keras.models import Sequential  

from keras.layers.core import Dense, Activation  

from keras.layers.recurrent import LSTM
train_x = np.array(df_train.iloc[:,2:])

train_t = np.array(df_train['target'])



length_of_sequences = 300

in_out_neurons = 1

hidden_neurons = 300



model = Sequential()  

model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  

model.add(Dense(in_out_neurons))  

model.add(Activation("linear"))  

model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.fit(train_x, train_t, batch_size=600, nb_epoch=15, validation_split=0.05) 

# from sklearn.ensemble import RandomForestClassifier

# rfc = RandomForestClassifier(random_state=0)



# rfc.fit(df_train.iloc[:,2:], df_train['target'])

# pred = rfc.predict(df_test.iloc[:,1:])



# submit_df = pd.DataFrame()

# submit_df['id'] = df_test['id']

# submit_df['target'] = pred

# submit_df.to_csv("submission.csv", index = False)
class LSTM(nn.Module):

    def __init__(self, seq_size, hidden_size, out_size):

        super(LSTM, self).__init__()

        self.xh = torch.nn.LSTM(seq_size, hidden_size)

        self.hy = torch.nn.Linear(hidden_size, out_size)

        self.hidden_size = hidden_size

    

    def __call__(self, xs):

        h, self.hidden = self.xh(xs, self.hidden)

        y = self.hy(h)

        return y

    

    def reset(self):

        self.hidden = (Variable(torch.zeros(1, 1, self.hidden_size)), Variable(torch.zeros(1, 1, self.hidden_size)))



EPOCH_NUM = 300

HIDDEN_SIZE = 5

BATCH_ROW_SIZE = 100

BATCH_COL_SIZE = 300

N = len(df_train)



train_x = np.array(df_train.iloc[:,2:])

train_t = np.array(df_train['target'])



# train_data = np.array([np.sin(i*2*np.pi/50) for i in range(50)]*10)

# train_x, train_t = [], []

# for i in range(0, len(train_data) - BATCH_COL_SIZE):

#     train_x.append(train_data[i:i+BATCH_COL_SIZE])

#     train_t.append(train_data[i+BATCH_COL_SIZE])

# train_x = np.array(train_x, dtype="float32")

# train_t = np.array(train_t, dtype="float32")



model = LSTM(seq_size=BATCH_COL_SIZE, hidden_size=HIDDEN_SIZE, out_size=1)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters())



print("Training ...")

st = datetime.datetime.now()

for epoch in range(EPOCH_NUM):

    x, t = [], []

    for i in range(BATCH_ROW_SIZE):

        index = np.random.randint(0, N)

        x.append(train_x[index])

        t.append(train_t[index])

    x = np.array(x, dtype="float64")

    t = np.array(t, dtype="float64")

    x = Variable(torch.from_numpy(x))

    t = Variable(torch.from_numpy(t))

    total_loss = 0

    model.reset()

    y = model(x)