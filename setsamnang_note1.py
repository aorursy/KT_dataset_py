# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
d = [1,2,3,4,5,6,7,8,9]  #define dataset
sc = MinMaxScaler(feature_range=(0,1))

d = np.array(d)
d = d.reshape(-1,1)
d.shape
d
dataset_scaled = sc.fit_transform(d)
dataset_scaled
sc.inverse_transform(dataset_scaled)
train_set = [1,2,3,4,5]

train_set = np.array(train_set)
train_set.shape
x_train = [[1,2],[2,3],[3,4],[4,5]]

x_train = np.array(x_train)
x_train.shape