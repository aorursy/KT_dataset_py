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
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col = 'Id')

train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col = 'Id')

train.head()
train.info()
drop_features = ['LotFrontage','Alley','FireplaceQu', 'PoolQC', 'MiscFeature', 'Fence']

train = train.drop(drop_features, axis=1)

train.info()
filteredColumns = train.dtypes[train.dtypes == 'object']

 

# list of columns whose data type is object i.e. string

objectdtype = list(filteredColumns.index)
df = train.copy()

for i in objectdtype:

    df[i] =df[i].astype('category').cat.codes
df.head()
x = df[df.columns[1:]].corr()['SalePrice'][:]
print(x.sort_values(ascending=False).head(35))
features_using = x.sort_values(ascending=False).head(26)
list_features = list(features_using.index.values)
new_df = df[list_features]

new_df.head(10)
new_df = new_df.drop('YearBuilt', axis=1)
new_df.describe()
new_df = new_df.drop('YearRemodAdd', axis=1)
new_df.info()
y = new_df['SalePrice']

new_df = new_df.drop('SalePrice', axis=1)
new_df.head()
new_df = new_df.drop('GarageYrBlt', axis=1)
new_df.head(10)
new_df.describe()
normalise_features = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'MasVnrArea', 'WoodDeckSF', 'BsmtFinSF1', '2ndFlrSF', 'OpenPorchSF', 'LotArea']
new = new_df.copy()

new.head()
for f in normalise_features:

    new[f] = (new[f]-new[f].min())/(new[f].max()-df[f].min())

new.head(10)
from sklearn.model_selection import train_test_split



X_train_full, X_valid_full, y_train, y_valid = train_test_split(new, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
from xgboost import XGBRegressor



# Define the model

my_model_1 = XGBRegressor(random_state=0) # Your code here



# Fit the model

my_model_1.fit(X_train_full, y_train)



from sklearn.metrics import mean_absolute_error



# Get predictions

predictions_1 = my_model_1.predict(X_valid_full)
# Calculate MAE

mae_1 = mean_absolute_error(y_valid, predictions_1) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_1)
my_model_2 = XGBRegressor(n_estimator=1000, learning_rate=0.05) # Your code here



# Fit the model

my_model_2.fit(X_train_full,y_train, early_stopping_rounds=20, eval_set=[(X_valid_full,y_valid)], verbose=False) # Your code here



# Get predictions

predictions_2 = my_model_2.predict(X_valid_full) # Your code here



# Calculate MAE

mae_2 = mean_absolute_error(y_valid, predictions_2) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)

new['SalePrice'] = y

new.head()
x = new[new.columns[1:]].corr()['SalePrice'][:]

print(x.sort_values(ascending=False))
remove_features = ['PavedDrive', 'Electrical', 'CentralAir', 'GarageQual']

df = new.drop(remove_features, axis=1)

df.head()
y = df['SalePrice']

X = df.drop('SalePrice', axis=1)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1,

                                                                random_state=0)

my_model_2 = XGBRegressor(n_estimator=1000, learning_rate=0.05) # Your code here



# Fit the model

my_model_2.fit(X_train_full,y_train, early_stopping_rounds=20, eval_set=[(X_valid_full,y_valid)], verbose=False) # Your code here



# Get predictions

predictions_2 = my_model_2.predict(X_valid_full) # Your code here



# Calculate MAE

mae_2 = mean_absolute_error(y_valid, predictions_2) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)
list_features.remove('SalePrice')

test_df = test[list_features]

test_df.head(10)
filteredColumns = test_df.dtypes[test_df.dtypes == 'object']

 

# list of columns whose data type is object i.e. string

objectdtype = list(filteredColumns.index)



df = test_df.copy()



for i in objectdtype:

    df[i] =df[i].astype('category').cat.codes

df.head()
new = df.copy()

for f in normalise_features:

    new[f] = (new[f]-new[f].min())/(new[f].max()-df[f].min())

new.head(10)
new = new.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1)
new.info()
new = new.drop(['PavedDrive', 'CentralAir', 'GarageQual', 'Electrical'], axis=1)
new.info()
preds_test = my_model_2.predict(new)
output = pd.DataFrame({'Id': new.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
X['SalePrice'] = y

X.head()
import matplotlib as plt

import seaborn as sns



sns.barplot(x = 'GarageCond', y = 'SalePrice', data = X)
sns.lmplot(x = 'LotArea', y = 'SalePrice', data = X)
last_remove = ['MasVnrArea', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'Foundation', 'GarageCond']

X = X.drop(last_remove, axis=1)

X = X.drop('SalePrice', axis=1)

X.head()
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

my_model_2 = XGBRegressor(n_estimator=1000, learning_rate=0.05) # Your code here



# Fit the model

my_model_2.fit(X_train_full,y_train, early_stopping_rounds=20, eval_set=[(X_valid_full,y_valid)], verbose=False) # Your code here



# Get predictions

predictions_2 = my_model_2.predict(X_valid_full) # Your code here



# Calculate MAE

mae_2 = mean_absolute_error(y_valid, predictions_2) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)
new = new.drop(last_remove, axis=1)

new.head()
preds_test = my_model_2.predict(new)



output = pd.DataFrame({'Id': new.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)