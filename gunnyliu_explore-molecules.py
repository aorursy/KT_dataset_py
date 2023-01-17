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



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping
DATA_PATH = "../input/"

train = pd.read_csv(DATA_PATH+"train.csv")

contributions = pd.read_csv(DATA_PATH+"scalar_coupling_contributions.csv")
train_contribs = train.merge(contributions, on=list(train.columns)[1:4])
assert(train_contribs.shape[0] == train.shape[0])

assert(contributions.shape[0] == train_contribs.shape[0])

assert(train_contribs.shape[1] == train.shape[1] + contributions.shape[1] - 3)
train_contribs.head(5)
# calc cov between each individual comp and scc

SAMPLING_RATE = 0.01

sample_idx = np.random.choice(np.arange(train_contribs.shape[0]), size=int(train_contribs.shape[0]*SAMPLING_RATE))

cov = np.cov(train_contribs[["scalar_coupling_constant", "fc", "sd", "dso", "pso"]].iloc[sample_idx].values.transpose())
sns.heatmap(cov)
corr = np.corrcoef(train_contribs[["scalar_coupling_constant", "fc", "sd", "dso", "pso"]].iloc[sample_idx].values.transpose())
sns.heatmap(corr, cmap="YlGnBu")

# fc, then dso best
enc = preprocessing.LabelEncoder()

enc.fit_transform(train.type)

sns.set(style="whitegrid")

sns.violinplot(x="type", y="scalar_coupling_constant", data=train)
# fit a 1-NN based on sample mean across bond types

SAMPLE_SIZE = 100

np.random.seed(2019)

type_sample_mean = {}

for t in train.type.unique():

    sample_idx = np.random.choice(train.loc[train["type"] == t].id, SAMPLE_SIZE)

    type_sample_mean[t] = np.mean(train.iloc[sample_idx].scalar_coupling_constant)

print("Lookup dict for type-based sample mean is:", type_sample_mean)

test = pd.read_csv(DATA_PATH+"test.csv")

test["scalar_coupling_constant"] = test.type.apply(lambda x: type_sample_mean[x])

test[["id", "scalar_coupling_constant"]].to_csv("tsm_submission.csv", index=False)
train_contribs.type_x = enc.fit_transform(train_contribs.type_x)
# example uses 3 layers, 200 each for 500 data points. overfit. need 10e4x params each W to get same overfit on mine.

NUNITS = 2000

NCOVARIATES = 5

COVARIATES = ["type_x", "fc", "sd", "dso", "pso"]

NLAYERS = 2

NEPOCHS = 1

P_SETOUT = 0.2

SEED = 2019

X, y = train_contribs[COVARIATES].values, train_contribs.scalar_coupling_constant.values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=P_SETOUT, random_state=SEED)

m = Sequential()

m.add(Dense(NUNITS, activation='relu', input_shape=(NCOVARIATES,)))

for _ in range(NLAYERS):

    m.add(Dense(NUNITS, activation='relu'))

m.add(Dense(1))

m.compile(optimizer='adam', loss='mean_squared_error')

m.fit(Xtrain, ytrain, 

      validation_split=P_SETOUT, epochs=NEPOCHS)
print(m.metrics_names)

print(m.evaluate(Xtest, ytest, verbose=1))