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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.describe()
print (train.shape)

train.head()
print (test.shape)

test.head()
from sklearn.linear_model import LinearRegression
missing_val = train.isnull().sum()

missing_val_col = missing_val[missing_val>0]

missing_val_col
train[missing_val_col.index.values].dtypes

train["LotFrontage"].replace(np.nan, train["LotFrontage"].mean(), inplace = True)
train["MasVnrArea"].replace(np.nan, train["MasVnrArea"].mean(), inplace = True)

train["GarageYrBlt"].replace(np.nan, train["GarageYrBlt"].mean(), inplace = True)

for col in missing_val_col.index.values:

    train[col].replace(np.nan, train[col].mode()[0], inplace = True)
missing_val_test = test.isnull().sum()

missing_val_test_col = missing_val_test[missing_val_test>0]

missing_val_test_col
for col in missing_val_test_col.index.values:

    if test[col].dtypes == "float":

        test[col].replace(np.nan, test[col].mean(), inplace = True)

    else:

        test[col].replace(np.nan, test[col].mode()[0], inplace = True)





train.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis = 1, inplace = True)

test.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis = 1, inplace = True)
train.shape, test.shape
fig = plt.figure()

ax0 = fig.add_subplot(121)

ax1 = fig.add_subplot(122)

sns.heatmap(train.isnull(), ax = ax0)

sns.heatmap(test.isnull(), ax = ax1)

## no missing values
test_id = test["Id"].to_list()

test_id
train.drop(["Id"], axis = 1, inplace = True)

test.drop(["Id"], axis = 1, inplace = True)
columns = train.select_dtypes(include = ["object"]).columns
df = pd.concat([train, test], axis = 0)

print (train.shape, test.shape, df.shape)

for col in columns:

    dummies = pd.get_dummies(df[col])

    df = pd.concat([df, dummies], axis = 1)

    df.drop([col], axis = 1, inplace = True)
df = df.loc[:, ~df.columns.duplicated()]

df.shape
df_train = df.iloc[:1460, :]

df_test = df.iloc[1460:, :]

df_test.drop(["SalePrice"], axis = 1, inplace = True)
df_test.shape, df_train.shape
Xtrain = df_train.drop(["SalePrice"], axis = 1)

Ytrain = df_train["SalePrice"]
import xgboost
model = xgboost.XGBRegressor()

model.fit(Xtrain, Ytrain)

predictions = model.predict(Xtrain)



ax1 = sns.distplot(Ytrain, hist = False, color = "r", label = "Actual")

sns.distplot(predictions, hist = False, color = "b", label = "Fitted", ax = ax1)
test_predict = model.predict(df_test)
submission = pd.DataFrame({"Id": test_id, "SalePrice": test_predict})

submission.head()
submission.to_csv("submission.csv", index = False)