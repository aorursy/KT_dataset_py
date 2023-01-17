# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#bring in the six packs

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# check the data

df_train.head()
#check the decoration

df_train.columns
#descriptive statistics summary

df_train['SalePrice'].describe()
#histogram

sns.distplot(df_train['SalePrice']);
# check the number of missing data

df_train.isnull().sum()
# string label to categorical values

from sklearn.preprocessing import LabelEncoder



for i in range(df_train.shape[1]):

    if df_train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(df_train.iloc[:,i].values) + list(df_test.iloc[:,i].values))

        df_train.iloc[:,i] = lbl.transform(list(df_train.iloc[:,i].values))

        df_test.iloc[:,i] = lbl.transform(list(df_test.iloc[:,i].values))
# example

print(df_train["SaleCondition"].unique())
# search for missing data

import missingno as msno

msno.matrix(df=df_train, figsize=(20,14), color=(0.5,0,0))
# keep ID for submission

train_ID = df_train['Id']

test_ID = df_test['Id']



# split data for training

y_train = df_train['SalePrice']

X_train = df_train.drop(['Id','SalePrice','LotFrontage', 'GarageYrBlt', 'MasVnrArea'], axis=1)



# normalization of X_train

import scipy.stats as sp

X_train = pd.DataFrame(sp.zscore(X_train), index=X_train.index, columns=X_train.columns)



X_test = df_test.drop(['Id','LotFrontage', 'GarageYrBlt', 'MasVnrArea'], axis=1)



# replace X_test missing data with mean data

X_test["GarageArea"] = X_test["GarageArea"].fillna(X_test["GarageArea"].mean())

X_test["GarageCars"] = X_test["GarageCars"].fillna(X_test["GarageCars"].mean())

X_test["BsmtFinSF1"] = X_test["BsmtFinSF1"].fillna(X_test["BsmtFinSF1"].mean())

X_test["BsmtUnfSF"] = X_test["BsmtUnfSF"].fillna(X_test["BsmtUnfSF"].mean())

X_test["TotalBsmtSF"] = X_test["TotalBsmtSF"].fillna(X_test["TotalBsmtSF"].mean())

X_test["BsmtFinSF2"] = X_test["BsmtFinSF2"].fillna(X_test["BsmtFinSF2"].mean())

X_test["BsmtHalfBath"] = X_test["BsmtHalfBath"].fillna(X_test["BsmtHalfBath"].mean())

X_test["BsmtFullBath"] = X_test["BsmtFullBath"].fillna(X_test["BsmtFullBath"].mean())



# normalization of X_test

X_test= pd.DataFrame(sp.zscore(X_test), index=X_test.index, columns=X_test.columns)



# show info

X_test.info()


X_train["TotalSF"] = X_train["TotalBsmtSF"] + X_train["1stFlrSF"] + X_train["2ndFlrSF"]

X_test["TotalSF"] = X_test["TotalBsmtSF"] + X_test["1stFlrSF"] + X_test["2ndFlrSF"]

X_test.info()
# log-transform the dependent variable for normality 



y_train = np.log(y_train)

ax = sns.distplot(y_train)

plt.show
#feature importance using random forest



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=80, max_features='auto')

rf.fit(X_train, y_train)

print('Training done using Random Forest')



ranking = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
# use the top 30 features only

X_train = X_train.iloc[:,ranking[:30]]

X_test = X_test.iloc[:,ranking[:30]]



# interaction between the top 2

X_train["Interaction"] = X_train["TotalSF"]*X_train["OverallQual"]

X_test["Interaction"] = X_test["TotalSF"]*X_test["OverallQual"]
# relation to the target

fig = plt.figure(figsize=(12,7))

for i in np.arange(30):

    ax = fig.add_subplot(5,6,i+1)

    sns.regplot(x=X_train.iloc[:,i], y=y_train)



plt.tight_layout()

plt.show()
# outlier deletion

Xmat = X_train

Xmat['SalePrice'] = y_train



Xmat = Xmat.drop(Xmat[(Xmat['TotalSF']>5) & (Xmat['SalePrice']<12.5)].index)

Xmat = Xmat.drop(Xmat[(Xmat['GrLivArea']>5) & (Xmat['SalePrice']<13)].index)



# recover

y_train = Xmat['SalePrice']

X_train = Xmat.drop(['SalePrice'], axis=1)
# XGBoost

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(X_train, y_train)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



def create_model(optimizer='adam'):

    model = Sequential()

    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))



    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model



model = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters

optimizer = ['SGD','Adam']

batch_size = [10, 30, 50]

epochs = [10, 50, 100]

param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)

reg_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

reg_dl.fit(X_train, y_train)
# SVR

from sklearn.svm import SVR



reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,

                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],

                               "gamma": np.logspace(-2, 2, 5)})

reg_svr.fit(X_train, y_train)
# second feature matrix

X_train2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_train),

     'DL': reg_dl.predict(X_train).ravel(),

     'SVR': reg_svr.predict(X_train),

    })



X_train2.head()
# second-feature modeling using linear regression

from sklearn import linear_model



reg = linear_model.LinearRegression()

reg.fit(X_train2, y_train)



# prediction using the test set

X_test2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_test),

     'DL': reg_dl.predict(X_test).ravel(),

     'SVR': reg_svr.predict(X_test),

    })



# Don't forget to convert the prediction back to non-log scale

y_pred = np.exp(reg.predict(X_test2))



# submission

submission = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": y_pred

})

submission.to_csv('houseprice.csv', index=False)



print("Done")