# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading the data
data = pd.read_csv('/kaggle/input/3dprinter/data.csv')

data.head()
data.info()
#Converting objects into integers
data.infill_pattern = [0 if each =="grid" else 1 for each in data.infill_pattern]
data.material = [0 if each =="abs" else 1 for each in data.material]
data.head()
data.info()
#converting the fractions into percentages
data.layer_height = data.layer_height * 100
data.elongation = data.elongation * 100 
x = data.drop(["material"], axis = 1) #remove materials column from dataset
y = data.material.values #giving y all the values of materials
#normalization and feature scaling
x_norm = (x-x.mean())/(x.std()) 
#x_norm = (x - avg_val_of_each_column) / (std_dev_of_each_column)

#Spliting data into three: train, cross validate, test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, random_state=1) 
    # 20% data -> x_test

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) 
# 0.25 x 0.8 = 0.2 -> 20% -> x_val

# x_train -> 60%, x_val -> 20%, x_test -> 20%
#Linear Regression
from sklearn.linear_model import LinearRegression

logreg = LinearRegression().fit(x_train, y_train)

#checking how much does model created using x_train fits the respective y_train values
#not that significant
logreg.score(x_train, y_train)
#predicted value of y using values of x_val
y_pred = logreg.predict(x_val)

#comparing the y_pred with the real labels
#y_pred is created automatically by .score() using x_val

logreg.score(x_val, y_val)
test_logreg = LinearRegression().fit(x_test, y_test)
logreg.score(x_test, y_test)
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, x_train, x_val, y_train, y_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(x_train, y_train)
    pred_val = model.predict(x_val)
    mae = mean_absolute_error(y_val, pred_val)
    return(mae)


#dt_model = DecisionTreeRegressor(random_state=1)
#dt_model.fit(x_train, y_train)

candidate_max_leaf_nodes = [2, 5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes,  x_train, x_val, y_train, y_val)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#y_pred_dt = dt_model.predict(x_val)     
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(x_test, y_test)
y_pred_dt = dt_model.predict(x_test)
val_mae = mean_absolute_error(y_test, y_pred_dt)
print(val_mae)