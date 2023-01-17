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
train.head()
test.tail()
train.isnull().sum().sum()
test.isnull().sum().sum()
X = train.drop(['Cover_Type'], axis = 1)

y = train.Cover_Type
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
forest_model = DecisionTreeRegressor(random_state=0)

forest_model.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error



val_predictions = forest_model.predict(X_val)

val_mae = mean_absolute_error(y_val,val_predictions)

val_mae
test_preds = forest_model.predict(test)

output = pd.DataFrame({'Id': test.Id, 'Cover_Type': test_preds.astype(int)})

output.to_csv('submission.csv', index=False)