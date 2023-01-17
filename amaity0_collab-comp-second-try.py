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
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
train.dtypes

test.dtypes
inp = train.iloc[:,15:-2].to_string(header=False, index=False, index_names=False).split('\n')

vals = [int(''.join(ele.split()),2) for ele in inp]

train['Soil_Type'] = vals

train.drop(train.iloc[:,15:-2], axis=1, inplace=True)

inp2 = train.iloc[:,11:15].to_string(header=False, index=False, index_names=False).split('\n')

vals2 = [int(''.join(ele.split()),2) for ele in inp2]

train['Wilderness_Area'] = vals2

train.drop(train.iloc[:,11:15], axis=1, inplace=True)

train.head()
X = train.drop(['Cover_Type'], axis = 1)

y = train.Cover_Type
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
forest_model = RandomForestClassifier(n_estimators=10)

forest_model.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error



val_predictions = forest_model.predict(X_val)

val_mae = mean_absolute_error(y_val,val_predictions)

val_mae
test['Soil_Type'] = test[test.columns[15:-1]].astype(str).apply(''.join,1).apply(int, base=2)

test.drop(test.iloc[:,15:-1], axis=1, inplace=True)

test['Wilderness_Area'] = test[test.columns[11:15]].astype(str).apply(''.join,1).apply(int, base=2)

test.drop(test.iloc[:,11:15], axis=1, inplace=True)



test.head()
test_preds = forest_model.predict(test)

output = pd.DataFrame({'Id': test.Id, 'Cover_Type': test_preds.astype(int)})

output.to_csv('submission.csv', index=False)