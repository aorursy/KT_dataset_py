# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np  # linear algebra 

import pandas as pd  # data processing, CSV file I / O (eg pd.read_csv)



# for visualization 

import matplotlib.pyplot as plt 

import seaborn as sns



# import data 

train=pd.read_csv('../input/train.csv') 

test=pd.read_csv('../input/test.csv')



#print(train.dtypes)
train['SalePrice'].describe()
# It seems we have nulls so we will use the imputer strategy later on.

Missing = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])

Missing[Missing.sum(axis=1) > 0]
train.info()
train.describe()
corr = train.corr()

plt.figure(figsize=(14,8))

plt.title('Overall Correlation of House Prices', fontsize=18)

sns.heatmap(corr,annot=False,cmap='BrBG',linewidths=0.2,annot_kws={'size':20})

plt.show()
# string label to categorical values

from sklearn.preprocessing import LabelEncoder



for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
print(train['SaleCondition'].unique())
# search for missing data

import missingno as msno

msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))
# keep ID for submission

train_ID = train['Id']

test_ID = test['Id']



# split data for training

y_train = train['SalePrice']

X_train = train.drop(['Id','SalePrice'], axis=1)

X_test = test.drop('Id', axis=1)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'LotFrontage', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    X_train[col] = X_train[col].fillna(0)

    X_test[col] = X_test[col].fillna(0)
Missing = pd.concat([X_train.isnull().sum(), X_test.isnull().sum()], axis=1, keys=['train', 'test'])

Missing[Missing.sum(axis=1) > 0]
#add a new feature 'total sqfootage'

X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']

X_test['TotalSF'] = X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']
#normality check for the target

ax = sns.distplot(y_train)

plt.show()
# log-transform the dependent variable for normality

y_train = np.log(y_train)

ax = sns.distplot(y_train)

plt.show()
# feature importance using random forest

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
sns.jointplot(x='GrLivArea',y='SalePrice',data=train,

              kind='hex', cmap= 'CMRmap', size=8, color='#F84403')



plt.show()
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')



fig, ax = plt.subplots(figsize=(14,8))

palette = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#FF8000", "#AEB404", "#FE2EF7", "#64FE2E"]



sns.swarmplot(x="OverallQual", y="SalePrice", data=train, ax=ax, palette=palette, linewidth=1)

plt.title('Correlation between OverallQual and SalePrice', fontsize=18)

plt.ylabel('Sale Price', fontsize=14)

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

##Xmat = X_train

#Xmat['SalePrice'] = y_train

#Xmat = Xmat.drop(Xmat[(Xmat['TotalSF']>5) & (Xmat['SalePrice']<12.5)].index)

#Xmat = Xmat.drop(Xmat[(Xmat['GrLivArea']>5) & (Xmat['SalePrice']<13)].index)



# recover

#y_train = Xmat['SalePrice']

#X_train = Xmat.drop(['SalePrice'], axis=1)
# XGBoost

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(X_train, y_train)
reg_xgb.score(X_train,y_train)
y_pred = reg_xgb.predict(X_train)

print(y_pred)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



def create_model(optimizer='adam'):

    model = Sequential()

    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

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
reg_dl.score(X_train,y_train)
y_pred = reg_dl.predict(X_train)

print(y_pred)
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
y_pred
# submission

submission = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": y_pred

})

submission.to_csv('houseprice.csv', index=False)
submission