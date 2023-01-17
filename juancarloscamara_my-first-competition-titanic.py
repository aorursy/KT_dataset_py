# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
input_train_path = '../input/train.csv'
input_test_path = '../input/test.csv'
input_train = pd.read_csv(input_train_path)
input_test = pd.read_csv(input_test_path)

print('====Format for the train file===')
input_train.info()

print('====Format for the test file===')
input_test.info()
print('====Header for the train file===')
input_train.head()
print('===Header for the test file===')
input_test.head()
# From the previous fields we see some things:
# - Cabin field is almost emptied in both files, train and test
# - Age field has Not valid values in both files, train and test, for some of the records
# - Fare is a field that potentially does not make sense to be related with the possibility of surviving or not - correlation necessary
# - Ticket is a field that potentially does not make sense to be related with the possibility of surviving or not - correlation necessary. However, not empty fields. We´ll keept it
# - Embarked is a field that potentially does not make sense to be related with the possibility of surviving or not - correlation necessary
# - Name is a field that could be divided and analyzed, but right now does not make sense to be related with survival
# - Sex is a field that make sense to analyzed, but it´s string. Conversion is necessary

# Let´s start aplying this changes in Train file

features = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch']
input_train_featured = input_train[features]

input_train_featured.head()

# Mapping Sex columnt to a series
Sex_map = {'male' : 0, 'female' : 1}
input_train_featured.Sex = input_train_featured.Sex.map(Sex_map)

input_train_featured.head()
# Extract Survived variable and featured dataset
input_train_featured_X = input_train_featured[['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch']]
input_train_featured_y = input_train_featured.Survived
# Now that we have a full train numeric dataset let´s proceed
# to create the models, fit, predict and see which one has a better MAE
from sklearn.model_selection import train_test_split

tmp_train_X, tmp_val_X, tmp_train_y, tmp_val_y = train_test_split(input_train_featured_X, input_train_featured_y, random_state = 5, test_size = 0.1)

print("=== Train X dataset for modeling ===")
tmp_train_X.info()
print("=== Val X dataset for modeling ===")
tmp_val_X.info()
# Let´s see with DecisionTreRegressor, which is the MAE, for the different max_leaf_nodes values

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 5)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
maes = []
nodes = [2,3,4,5,10,20,25,30,40, 50,75,100,200,250,300, 500,600,700,800,900,1000,1500,2000,2500,3000,4000, 5000]
for max_leaf_nodes in nodes:
    my_mae = get_mae(max_leaf_nodes, tmp_train_X, tmp_val_X, tmp_train_y, tmp_val_y)
    print("Max leaf nodes: %f  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))
    maes.append(my_mae)

    
print("Best index %d, best mae %f" %(nodes[maes.index(min(maes))], min(maes)))

# Let´s see with RandomForestRegressor, which is the MAE
from sklearn.ensemble import RandomForestRegressor

def get_mae_randomforest(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(random_state = 5)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

my_mae = get_mae_randomforest(tmp_train_X, tmp_val_X, tmp_train_y, tmp_val_y)
print("Mean Absolute Error:  %f" %( my_mae))
# It can be seen that MAE for Decision is better than one for RandomForest, for a given max_leaf_nodes attribute.
# We will proceed creating a new modeling for the whole dataset, not only 90%, and proceed with executing the estimation for the test.csv

final_model = DecisionTreeRegressor(max_leaf_nodes=250, random_state = 5)
final_model.fit(input_train_featured_X, input_train_featured_y)
# Not it´s time for applying the same changes we applied for the train dataset
test_features = ['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch']
input_test_featured = input_test[test_features]

Sex_map = {'male' : 0, 'female' : 1}
input_test_featured.Sex = input_test_featured.Sex.map(Sex_map)

input_test_featured.head()
predictions = final_model.predict(input_test_featured)
print(predictions)

my_submission = pd.DataFrame({'PassengerId': input_test_featured.PassengerId, 'Survived': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
# For some reason, I am receiving 0.041666... values in the dataset. I´m still investigating why, I didn´t expect them.
# As the values are quite small, I will assume they can be rounded to 0

predictions2 = predictions.round()
print(predictions2)
my_submission2 = pd.DataFrame({'PassengerId': input_test_featured.PassengerId, 'Survived': predictions2})
# you could use any filename. We choose submission here
my_submission2.to_csv('submission2.csv', index=False)
predictions3 = final_model.predict(input_test_featured).astype(int)
print(predictions3)

my_submission3 = pd.DataFrame({'PassengerId': input_test_featured.PassengerId, 'Survived': predictions3})
# you could use any filename. We choose submission here
my_submission3.to_csv('submission3.csv', index=False)
