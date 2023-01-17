import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
#loading Data



df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(df.shape)

print(df_test.shape)
#pd.options.display.max_columns = None

df.head(5)
df.describe()

# get only null columns

nullcol = df.columns[df.isna().any()]
df[nullcol].isnull().sum()
df[nullcol].isnull().sum() * 100 / len(df)
#dropping the columns where the missing values are over 40%



df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
#checking the data types of missing value columns

nullcol = df.columns[df.isna().any()]

df[nullcol].dtypes


#df.fillna(-999, inplace=True)

#df_test.fillna(-999, inplace=True)
# selecting columns where the type is strings with missing values



objcols= df[nullcol].select_dtypes(['object']).columns

objcols
#replacing the missing values of the strings with the -999



df[objcols] = df[objcols].replace(np.nan, -999)
df[objcols].isnull().sum()
#imputing numeric values



#get numerical features by dropping categorical features from the list

num_null=(nullcol.drop(objcols))



df[num_null] = df[num_null].fillna(df.mean().iloc[0])



df.columns[df.isna().any()]
#numerical data

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



num_cols = df.select_dtypes(include=numerics)



#categorical data

string_cols = df.select_dtypes(exclude =numerics)


print(num_cols.shape)

print(string_cols.shape)
test_nullcol = df_test.columns[df_test.isna().any()]

df_test[test_nullcol].isnull().sum() * 100 / len(df_test)
test_nullcol = df_test.columns[df_test.isna().any()]



#string columns in test 

test_objcols= df_test[test_nullcol].select_dtypes(['object']).columns





df_test[test_objcols] = df_test[test_objcols].replace(np.nan, -999)



#get numerical features by dropping categorical features from the list

test_num_null=(test_nullcol.drop(test_objcols))





#replacing with the mean

df_test[test_num_null] = df[test_num_null].fillna(df_test.mean().iloc[0])

df_test.columns[df_test.isna().any()]
#drop features that were removed from trainig set

df_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
df_test.shape
#numerical data

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



#numerical features in the test set

test_num_cols = df_test.select_dtypes(include=numerics)



#categorical data features

test_string_cols = df_test.select_dtypes(exclude =numerics)
print(df.shape)

print(df_test.shape)
#assiging X and target label y

y = df['SalePrice']

X = df.drop('SalePrice',axis=1)
print(y.shape)

print(X.shape)
categorical_features_indices = np.where(X.dtypes != np.float)[0]



#splitting data to training and testing

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=40)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from catboost import CatBoostRegressor



model=CatBoostRegressor(iterations=1200, depth=7, learning_rate=0.1, loss_function='RMSE')



model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))
print("Train Score", model.score(X_train,y_train))

print("Test Score", model.score(X_test,y_test))
