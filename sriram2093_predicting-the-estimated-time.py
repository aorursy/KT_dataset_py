# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

print(data.shape)
print(data.columns)

print(data.head(5))
#removing the ID column

data = data.ix[:, data.columns != 'ID']



#Splitting the target variable and other features

X = data.ix[:, data.columns != 'y']

y = data['y']

print(X.shape)

print(y.shape)
from sklearn.cross_validation import cross_val_score, ShuffleSplit

from sklearn.ensemble import RandomForestRegressor



names = data.columns