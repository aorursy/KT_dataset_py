# Code you have previously used to load data

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from learntools.core import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)

import seaborn as sns

import matplotlib.pyplot as plt
home_data.describe()

home_data.info()
sns.distplot(home_data['SalePrice'])
#f,ax=plt.subplots(figsize=(15,12))

#sns.heatmap(home_data.corr(),vmax=0.8,square=True)

k=10

cols=home_data.corr().nlargest(k,'SalePrice')['SalePrice']

cols
train_y=np.log1p(home_data.pop('SalePrice'))

#sns.distplot(train_y)

test_data=pd.read_csv('../input/test.csv')

train_data=pd.read_csv('../input/train.csv')

train_data.pop('SalePrice')

train_data.set_index(['Id'],inplace=True)

test_data.set_index(['Id'],inplace=True)

full_data=train_data.append(test_data)

#full_data.info()

#full_data['MSSubClass'].value_counts()

#pd.get_dummies(full_data['MSSubClass'],prefix='MSSubClass').head()

full_dummy_data=pd.get_dummies(full_data)

#full_dummy_data.isnull().sum().sort_values(ascending=False).head(15)

mean_cols=full_dummy_data.mean()

full_dummy_data=full_dummy_data.fillna(mean_cols)

full_dummy_data.isnull().sum()
numeric_cols=full_data.columns[full_data.dtypes!='object']

numeric_col_mean=full_dummy_data.loc[:,numeric_cols].mean()

numeric_col_std=full_dummy_data.loc[:,numeric_cols].std()

full_dummy_data.loc[:,numeric_cols]=(full_dummy_data.loc[:,numeric_cols]-numeric_col_mean)/numeric_col_std
dummy_train_data=full_dummy_data.loc[train_data.index]

dummy_test_data=full_dummy_data.loc[test_data.index]

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

alphas=np.logspace(-3,2,50)

X_train=dummy_train_data.values

X_test=dummy_test_data.values

test_scores=[]

for alpha in alphas:

    clf=Ridge(alpha)

    test_score=np.sqrt(-cross_val_score(clf,X_train,train_y,cv=10,scoring='neg_mean_squared_error'))

    test_scores.append(np.mean(test_score))

plt.plot(alphas,test_scores)

plt.title('alphas vs cv error')
N_estimators=[20,50,100,150,200,250,300]

test_scores=[]

for N in N_estimators:

    clf=RandomForestRegressor(n_estimators=N,max_features=0.3)

    test_score=np.sqrt(-cross_val_score(clf,X_train,train_y,cv=5,scoring='neg_mean_squared_error'))

    test_scores.append(np.mean(test_score))  

plt.plot(N_estimators,test_scores)

plt.title('N_estimator vs CV Error')



ridge=Ridge(alpha=15)

rf=RandomForestRegressor(n_estimators=350,max_features=0.3)

ridge.fit(X_train,train_y)

rf.fit(X_train,train_y)

ridge_predict=ridge.predict(X_test)

y_ridge=np.expm1(ridge_predict)

rf_predict=rf.predict(X_test)

y_rf=np.expm1(rf_predict)

y_final=(y_ridge+y_rf)/2


output = pd.DataFrame({'Id': test_data.index,

                       'SalePrice': y_final})

output.to_csv('submission.csv', index=False)