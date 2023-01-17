import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
My_Data= pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",index_col='Id')

My_Data.head()

My_Data.shape
# Remove rows with missing target, separate target from predictors



My_Data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = My_Data.SalePrice

My_Data.drop(['SalePrice'], axis=1, inplace=True)



My_Data.shape

# To keep things simple, we'll use only numerical predictors

X = My_Data.select_dtypes(exclude=['object'])



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)

X_train.shape
# Number of missing values in each column of training data

missing_by_column = (X_train.isnull().sum())

print(missing_by_column[missing_by_column > 0])
X_train.shape
# Fill in the line below: get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]





# Fill in the lines below: drop columns in training and validation data



reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
reduced_X_train.shape
from sklearn.tree import DecisionTreeRegressor

my_model = DecisionTreeRegressor(random_state=1)

my_model.fit(reduced_X_train, y_train)
from sklearn.metrics import mean_absolute_error

Predict = my_model.predict(reduced_X_valid)

val_mae = mean_absolute_error(Predict, y_valid)

print("Validation MAE: {:,.0f}".format(val_mae))
X_Test= pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv",index_col="Id")

X_Test.head()
X_Test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col='Id')
XT = X_Test.select_dtypes(exclude=['object'])

missing_by_column_test = (XT.isnull().sum())

print(missing_by_column_test[missing_by_column_test > 2])

cols_missing=['LotFrontage','MasVnrArea','GarageYrBlt']

reduced_X_test = XT.drop(cols_missing, axis=1)
cols_with_missing_T = [col for col in XT.columns

                     if XT[col].isnull().any()]





reduced_X_test.fillna(-1,inplace=True)
reduced_X_test.shape

reduced_X_test.head()

reduced_X_test.index
result=my_model.predict(reduced_X_test)

id=reduced_X_test.index

out=pd.DataFrame({'Id':id,'Salesprice':result})



out.to_csv('submission.csv', index=False)
