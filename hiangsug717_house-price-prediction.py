import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# back up

train = train_df.copy()

test = test_df.copy()
sns.set()

corrmat = train.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, square=True)
k  = 10 # Top 10 features correlated with SelePrice

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, \

                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Visualize the relationship between paired features

sns.set()

sns.pairplot(train[cols], height = 2.5)

plt.show()
train[cols].isnull().sum()
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

X = train[cols].iloc[:,1:].values

Y = train[cols].iloc[:,0].values

x_scaled = preprocessing.StandardScaler().fit_transform(X)

y_scaled = preprocessing.StandardScaler().fit_transform(Y.reshape(-1,1))

X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
y_train
from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings('ignore')





clfs = {

        'svm':svm.SVR(), 

        'BayesianRidge':linear_model.BayesianRidge(),

        'RandomForestRegressor':RandomForestRegressor(n_estimators=400)

       }



cost = np.zeros(len(clfs))

acc = np.zeros(len(clfs))

i = 0

for clf in clfs:

    clfs[clf].fit(X_train, y_train)

    y_pred = clfs[clf].predict(X_test)

    cost[i] = round(np.sum(y_pred-y_test)/len(y_pred),2)

    print(clf + " cost:" + str(cost[i]))

    acc[i] = round(clfs[clf].score(X_train, y_train) * 100, 2)

    print(clf + " acc:" + str(acc[i]))

    i = i+1
models = pd.DataFrame([clfs.keys(),cost,acc],index = ['model','cost','acc']).T

models
data_test = test[cols[1:]]
data_test.isnull().sum()
data_test['GarageCars'] = data_test['GarageCars'].fillna(data_test['GarageCars'].mean())

data_test['GarageArea'] = data_test['GarageArea'].fillna(data_test['GarageCars'].mean())

data_test['TotalBsmtSF'] = data_test['TotalBsmtSF'].fillna(data_test['TotalBsmtSF'].mean())

data_test.isnull().sum()
predictions = clfs['RandomForestRegressor'].predict(data_test)

predictions
submission = pd.DataFrame(test['Id'],columns = ['Id'])

submission['SalePrice'] = predictions

submission
submission.to_csv('submission.csv',index = False)