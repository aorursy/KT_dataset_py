# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def get_data():

    train_df = pd.read_csv('../input/train.csv')

    train_data = train_df.values

    train_features = np.float64(train_data[:, 2:])

    

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(train_features)

    train_features = scaler.transform(train_features)

    

    train_target = np.float64(train_data[:, 1])

    return train_features, train_target
train_features, train_target = get_data()
warnings.filterwarnings('ignore')
train_data = pd.DataFrame(train_features)

train_data['target'] = train_target

columns = train_data.columns



for column in columns:

    if column != 'target':

        sns.pairplot(x_vars=column, y_vars=column, hue='target', diag_kind='kde', data=train_data, palette=['green', 'red'])
zero_samples = []

one_samples = []



for i, target in enumerate(train_target):

    if target == 0:

        zero_samples.append(train_features[i])

    else:

        one_samples.append(train_features[i])



zero_samples = np.array(zero_samples)

one_samples = np.array(one_samples)
for sample in zero_samples[:100]:

    sns.heatmap(sample.reshape((8, 25)))

    plt.show()
for sample in one_samples[:100]:

    sns.heatmap(sample.reshape((8, 25)))

    plt.show()