# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/apndcts/apndcts.csv")
print(data.shape) 
#holdout

from sklearn.model_selection import train_test_split

data_train , data_test, = train_test_split(data, test_size =0.3, random_state = 123)

print(data_train.shape)

print(data_test.shape)
# Kfold

from sklearn.model_selection import KFold

kf = KFold(n_splits = 10)

for train_index, test_index in kf.split(data):

    data_train = data.iloc[train_index]

    data_test = data.iloc[test_index]

print(data_train.shape)

print(data_test.shape)
#bootstrap

from sklearn.utils import resample

X = data.iloc[:,0:9]

resample(X, n_samples = 5, random_state = 100)
