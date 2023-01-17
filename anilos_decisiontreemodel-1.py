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
train_file_path = '../input/home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)
train_data.columns
train_data.describe()
train_data.head()
train_data.columns
y = train_data.SalePrice
X = X.select_dtypes(exclude=['object'])
X.describe()
X.columns
# Test Data Missing Values



# (1459, 33)

# BsmtFinSF1      1

# BsmtFinSF2      1

# BsmtUnfSF       1

# TotalBsmtSF     1

# BsmtFullBath    2

# BsmtHalfBath    2

# GarageCars      1

# GarageArea      1

# dtype: int64
print(X.shape)
train_data_features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',

       'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',

       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold']
X = train_data[train_data_features]
X.head()
X.describe()
from sklearn.tree import DecisionTreeRegressor

train_data = DecisionTreeRegressor()
s = (X.dtypes == 'object')

object_cols = list(s[s].index)

object_cols
print("Categorical variables:")

print(object_cols)
X = X.select_dtypes(exclude=['object'])

X
X.head()
X.describe()
from sklearn.tree import DecisionTreeRegressor
train_model = DecisionTreeRegressor(random_state=1)
train_model.fit(X,y)
print("Making predictions for the following 5 houses:")

print(X.head())
print("The predictions are")

print(train_model.predict(X.head()))
train_model.predict(X)
y
from sklearn.metrics import mean_absolute_error
predicted_train_model = train_model.predict(X)
mean_absolute_error(predicted_train_model,y)
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)
tain_model = DecisionTreeRegressor(random_state=1)
train_model.fit(train_X,train_y)
val_predictions = train_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print('Validation MAE: {:,.0f}'.format(val_mae))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
canditate_max_leaf_nodes = [5, 10, 25, 35, 50, 65, 80, 100, 250, 500, 1000]
# A loop to find the ideal tree size from canditate_max_leaf_nodes

for max_leaf_nodes in canditate_max_leaf_nodes:

    my_mae = get_mae(max_leaf_nodes, val_X, train_X, val_y, train_y)

    print('mae: ',  max_leaf_nodes, my_mae) 
final_model = DecisionTreeRegressor(max_leaf_nodes=50, random_state=1)
print(final_model)
final_model.fit(X,y)
y
print(final_model)
y
X
y
output = pd.DataFrame({'Id': train_data.Id,

                       'SalePrice': y})

output.to_csv('submission.csv', index=False)
test_file_path = '../input/home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_file_path)
test_data.describe()
test_data
test_X = test_data[train_data_features]
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

imputed_test_X = pd.DataFrame(my_imputer.fit_transform(test_X))

imputed_test_X.columns = test_X.columns




print(imputed_test_X)



missing_val_count_by_column = (imputed_test_X.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
predict_test = final_model.predict(imputed_test_X)
print(predict_test)
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predict_test}) 

my_submission.to_csv('submission.csv', index = False)