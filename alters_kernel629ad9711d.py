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
travaller_data_path = os.path.join(dirname,filename)

travaller_data = pd.read_csv(travaller_data_path)

y = travaller_data.Attrition

travaller_data_features = ['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education']

X = travaller_data[travaller_data_features]

from sklearn.model_selection import train_test_split

train_X, val_X, train_y,val_y = train_test_split(X,y,random_state = 0)
one_hot_encoded_train_X = pd.get_dummies(train_X)

one_hot_encoded_train_y = pd.get_dummies(train_y)

one_hot_encoded_val_X = pd.get_dummies(val_X)

one_hot_encoded_val_y = pd.get_dummies(val_y)

from sklearn.tree import DecisionTreeRegressor

my_model = DecisionTreeRegressor(random_state = 1)

my_model.fit(one_hot_encoded_train_X,one_hot_encoded_train_y)
from sklearn.metrics import mean_absolute_error

val_pred = my_model.predict(one_hot_encoded_val_X)

print(mean_absolute_error(one_hot_encoded_val_y,val_pred))
def get_mae(max_leaf_nodes,one_hot_encoded_train_X,one_hot_encoded_val_X,one_hot_encoded_train_y, one_hot_encoded_val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(one_hot_encoded_train_X, one_hot_encoded_train_y)

    preds_val = model.predict(one_hot_encoded_val_X)

    mae = mean_absolute_error(one_hot_encoded_val_y, preds_val)

    return(mae)
for max_leaf_nodes in range(2,100,10):

    my_mae = get_mae(max_leaf_nodes,one_hot_encoded_train_X,one_hot_encoded_val_X,one_hot_encoded_train_y, one_hot_encoded_val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))