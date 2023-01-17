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



import matplotlib.pyplot as plt

%matplotlib inline

import sklearn.preprocessing

from keras.models import Sequential

from keras.layers import Dense

from sklearn import linear_model

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

import seaborn as sns

sns.set()



train = pd.read_csv('../input/train.csv', index_col = 0)

test = pd.read_csv('../input/test.csv', index_col = 0)

full = pd.concat([train,test],axis=1)
# Sort columns by number of na and missing values to be imputed later

train = pd.get_dummies(train)

cols_with_missing = [col for col in train.columns if train[col].isnull().any()]



my_imputer = SimpleImputer()

imputed_data = my_imputer.fit_transform(train[cols_with_missing])

imputed_data = pd.DataFrame(imputed_data)

imputed_data.columns = train[cols_with_missing].columns

train[cols_with_missing] = imputed_data



train['yearsToSell'] = train['YrSold'] - train['YearBuilt']

train['yearsSinceRemod'] = train['YrSold'] - train['YearRemodAdd']
sns.scatterplot(data=train, x=train.index, y='SalePrice')
train.columns
sns.set_context("notebook", font_scale=1.1)

sns.set_style("ticks")

cols_to_see = ['SalePrice','LotArea','TotRmsAbvGrd','GrLivArea', 'yearsToSell','yearsSinceRemod']

sns.pairplot(data=train[cols_to_see])
sns.lmplot(data=train,x='GrLivArea',y='SalePrice')
sns.regplot(data=train,x='YearBuilt',y='SalePrice')
sns.regplot(data=train,x='yearsToSell',y='SalePrice')
sns.distplot(train['YearBuilt'], norm_hist=True)
train['hasBsmt'] = np.where(train['TotalBsmtSF'] > 0, 1, 0)

train['hasPool'] = np.where(train['PoolArea'] > 0, 1, 0)

train['totalArea'] = train['GrLivArea'] + train['GarageArea'] + train['PoolArea']
sns.distplot(train['totalArea'])
train['hasPool'].sum()
#correlation matrix

corrmat = train.corr()
# pmarcelino has a great way of clarifying the coefficients with this code. I plan on using this often

#saleprice correlation matrix

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
y = train['SalePrice']

x = train.drop(labels='SalePrice', axis=1)

x = x.fillna(0)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)

x_scaled = pd.DataFrame(x_scaled, columns=x.columns)



x = x_scaled

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)
print(xtrain.shape)

print(ytrain.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



model = RandomForestRegressor(random_state=17,n_estimators=1000)

model.fit(xtrain,ytrain)

preds = model.predict(xtest)

print(mean_absolute_error(ytest, preds))

rf_results = mean_absolute_error(ytest, preds)
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(xtrain,ytrain)

preds = model.predict(xtest)

xgb_results = mean_absolute_error(preds,ytest)

print("MAE : " + str(mean_absolute_error(preds,ytest)))

my_submission = pd.DataFrame({'Id': ytest.index, 'SalePrice' : ytest})

my_submission.to_csv('submission.csv', index=False)
reg = linear_model.Lasso(alpha=0.3, normalize=True)

reg.fit(xtrain,ytrain)

lasso_pred = reg.predict(xtest)

mse = np.mean((lasso_pred - ytest)**2)

reg.score(xtest,ytest)
from sklearn.linear_model import LassoCV

reg = LassoCV(cv=50, random_state=17).fit(xtrain,ytrain)

reg.score(xtest,ytest)



from sklearn.linear_model import ElasticNet

net = ElasticNet(alpha=0.1,l1_ratio=1,random_state=17)

net.fit(xtrain,ytrain)

net.score(xtest,ytest)
from sklearn import svm

clf = svm.SVR()

clf.fit(xtrain,ytrain)

svm_preds = clf.predict(xtest)

rmse = np.sqrt(np.mean((svm_preds - ytest)**2))

rmse
from sklearn.neural_network import MLPRegressor

neural_net = MLPRegressor(hidden_layer_sizes=(20,20),activation='relu',alpha=2, random_state=17)

neural_net.fit(xtrain,ytrain)

nn_preds = neural_net.predict(xtest)

neural_net.score(xtest,ytest)