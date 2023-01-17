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
#Library



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train['train']  = 1

test['train']  = 0

df = pd.concat([train, test], axis=0,sort=False)
df.head()
train.head()
train.tail()
df.describe()
df.isnull().values.any()
df.info()
#correlation matrix

corrmat = df.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, annot=True);
# most correlated features

corrdf = df.corr()

max_corr_features = corrdf.index[abs(corrdf["SalePrice"])>0.6]

plt.figure(figsize=(10,10))

sns.heatmap(df[max_corr_features].corr(),annot=True)
fig, axis = plt.subplots(1, 3,figsize=(12,6))

sns.regplot(x = 'OverallQual', y = 'SalePrice', data=df,ax=axis[0])

sns.regplot(x = 'GrLivArea', y = 'SalePrice', data=df,ax=axis[1])

sns.regplot(x = 'GarageCars', y = 'SalePrice', data=df,ax=axis[2])
df_train_target = df[df['train'] == 1]['SalePrice']

df = df.drop(['SalePrice'],axis=1)
df.columns
# Split numerical and categorical features



categorical_features = df.select_dtypes(include = ["object"]).columns

numerical_features = df.select_dtypes(exclude = ["object"]).columns

#numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

df_numerical = df[numerical_features]

df_category = df[categorical_features]
# Compute missing values for numerical features and fill them with by median as replacement

print("Number of NAs for numerical features in data: " + str(df_numerical.isnull().values.sum()))
df_numerical = df_numerical.fillna(df_numerical.median())
df_category = pd.get_dummies(df_category)

print("train category",df_category.shape)

print("train numerical",df_numerical.shape)

df_new = pd.concat([df_category,df_numerical],axis=1)

df_new.shape


df_train = df_new[df_new['train'] == 1]

df_train = df_train.drop(['train',],axis=1)





df_test = df_new[df_new['train'] == 0]

df_test = df_test.drop(['train',],axis=1)
y = df_train_target
#split the data to train the model 

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(df_train,y,test_size = 0.3,random_state= 0)
print("X_train",X_train.shape)

print("X_test",X_test.shape)

print("y_train",y_train.shape)

print("y_test",y_test.shape)

from xgboost import XGBRegressor



xgb =XGBRegressor( booster='gbtree',colsample_bytree=0.5, learning_rate=0.01, max_delta_step=0,

             max_depth=4, min_child_weight=1, n_estimators=1000,

             n_jobs=1, nthread=None, objective='reg:linear',

             reg_alpha=0.5, reg_lambda=0.5, scale_pos_weight=1, 

             silent=None, subsample=0.7, verbosity=1)

xgb.fit(X_train, y_train)
predict = xgb.predict(X_test)
import math

import sklearn.metrics as metrics



print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))

X_train.shape
#plt.scatter(X_train,y_train, color='blue')
xgb.fit(df_train, df_train_target)
predict2 = xgb.predict(df_test)

predict_y = ( predict2)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)





coeff_df = pd.DataFrame(lr.coef_,df_train.columns,columns=['Coefficient'])

coeff_df
print(coeff_df.idxmax().values[0])

print(coeff_df.idxmin().values[0])
lr.score(X_test,y_test)

print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predict_y

    })

submission.to_csv('submission.csv', index=False)