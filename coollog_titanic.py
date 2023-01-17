# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_all = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Split data into 80% train 20% validation
split_data_mask = np.random.rand(len(train_all)) < 0.8
train = train_all[split_data_mask]
valid = train_all[~split_data_mask]
train.dtypes
# Select data to fit
data = train.loc[:,['Age']]
target = train.loc[:, 'Survived']
valid_data = valid.loc[:,['Age']]
valid_target = valid.loc[:, 'Survived']

# Remove all NaN
nan_mask = np.any(np.isnan(data), axis=1)
data = data[~nan_mask]
target = target[~nan_mask]

valid_nan_mask = np.any(np.isnan(valid_data), axis=1)
valid_data = valid_data[~valid_nan_mask]
valid_target = valid_target[~valid_nan_mask]
# Fit the data
gnb = GaussianNB()
gnb_model = gnb.fit(data, target)
gnb_pred = gnb_model.predict(valid_data)
# Get predictive accuracy
np.sum(gnb_pred == valid_target) / len(valid_target)