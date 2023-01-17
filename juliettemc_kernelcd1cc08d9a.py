# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_path = "../input/train.csv"
titanic_data = pd.read_csv(train_path)

titanic_data.head()
#titanic_data.describe()
#Specify target and features
y = titanic_data.Survived

features=["Pclass","Age"]   # feature Sex a ajouter apres Level 2 car non numerical / feature Age aussi car missing values
X = titanic_data[features]
missing_val_count_by_column = X.isnull().sum()

print(missing_val_count_by_column)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
#print(train_X.head())
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]

#Imputation
imputed_train_X = train_X.copy()
imputed_val_X = val_X.copy()

for col in cols_with_missing:
    imputed_train_X[col + '_was_missing'] = imputed_train_X[col].isnull()
    imputed_val_X[col + '_was_missing'] = imputed_val_X[col].isnull()
    
#print(imputed_train_X)

my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(imputed_train_X))
imputed_val_X = my_imputer.transform(imputed_val_X)

#print(imputed_train_X)

#Model tree
titanic_treeModel = DecisionTreeRegressor(max_leaf_nodes=5000, random_state=1)
titanic_treeModel.fit(imputed_train_X, train_y)

tree_predictions = titanic_treeModel.predict(imputed_val_X)
tree_preds = np.round(tree_predictions)
tree_mae = mean_absolute_error(tree_preds, val_y)

print(tree_mae)

#Model Forest
titanic_forestModel = RandomForestRegressor(random_state=1)
titanic_forestModel.fit(imputed_train_X, train_y)

forest_predictions = titanic_forestModel.predict(imputed_val_X)
forest_preds = np.round(forest_predictions)
forest_mae = mean_absolute_error(forest_preds,val_y)

print(forest_mae)
#Imputation on full data
full_imputer = SimpleImputer()
imputed_X = full_imputer.fit_transform(X)

#Model Forest on full data
fulldata_forestModel = RandomForestRegressor(random_state=1)
fulldata_forestModel.fit(imputed_X,y)

#read test file and predict on it
test_path = "../input/test.csv"
test_data = pd.read_csv(test_path)
test_X = test_data[features]
imputed_test_X = full_imputer.transform(test_X)

fulldata_predictions = fulldata_forestModel.predict(imputed_test_X)
fulldata_preds = np.round(fulldata_predictions).astype(int)
#print(fulldata_predictions)
#print(fulldata_preds)
#generate output
output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':fulldata_preds})
output.to_csv('submission.csv', index = False)
print(output)