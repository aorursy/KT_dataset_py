# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
stacked = pd.concat([train,test])
from sklearn.preprocessing import LabelEncoder

for col in stacked.columns:

    if stacked[col].dtype == 'O':

        stacked = pd.concat
mean = np.mean(train, axis=0)

mean['SalePrice'] = np.nan

stacked = stacked.fillna(mean)
X_train = stacked[~np.isnan(stacked.SalePrice)].drop('SalePrice', 1)

X_test = stacked[np.isnan(stacked.SalePrice)].drop('SalePrice', 1)

y_train = stacked[~np.isnan(stacked.SalePrice)]['SalePrice']
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(verbose=2, n_jobs=-1)

model.fit(X_train, y_train)

y_test = model.predict(X_test)
help(le)