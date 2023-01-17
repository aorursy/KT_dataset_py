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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head(10)
test.head()
train['MSZoning'].value_counts()
sns.heatmap(train.isnull(),cbar = False)
train.shape
train.info()
## fill Missing Values

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
#verify if value is filled in 

train.info()
train.drop(['Alley'], axis = 1, inplace = True)
print(train['BsmtCond'].mode()[0])
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train['FireplaceQu'] = train['FireplaceQu'].fillna('NA')

train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train['GarageYrBlt'] = 2020 - train['GarageYrBlt']
train['GarageYrBlt'].unique()
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train.drop(['PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
train.shape
sns.heatmap(train.isnull(), cbar = False)
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

train['BsmtFinSF1'] = train['BsmtFinSF1'].fillna(train['BsmtFinSF1'].mean())
for str in train.columns:

    if train[str].isnull().sum() != 0:

        print(str)

        
print(train['BsmtExposure'].unique())
train['BsmtExposure'] = train['BsmtExposure'].fillna('NA')
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NA')

train['BsmtFinType2'] = train['BsmtFinType2'].fillna('NA')
train['Electrical'] = train['Electrical'].fillna('Mix')
for str in train.columns:

    if train[str].isnull().sum() != 0:

        print(str)
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
sns.heatmap(train.isnull(), cbar = False)
train.info()

list_col_object = []

for str in train.columns:

    if train[str].dtype == 'O':

        list_col_object.append(str)

print(list_col_object)

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



#encode the sex column

for str in list_col_object:

    train[str] = labelencoder.fit_transform( train[str].values)

    print(train[str].unique())
X = train.iloc[:, 1:75].values

Y = train['SalePrice'].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
def models(X_train, Y_train):



  #logistic regression

  from sklearn.linear_model import LogisticRegression

  log = LogisticRegression(random_state = 0)

  log.fit(X_train, Y_train)



  #Knn

  from sklearn.neighbors import KNeighborsClassifier

  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)

  knn.fit(X_train, Y_train)

  

  #SVC

  from sklearn.svm import SVC

  svc_lin = SVC(kernel = 'linear', random_state = 0)

  svc_lin.fit(X_train, Y_train)



  #Gaussian NB

  from sklearn.naive_bayes import GaussianNB

  gauss = GaussianNB()

  gauss.fit(X_train, Y_train)



  #Descision Tree

  from sklearn.tree import DecisionTreeClassifier

  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

  tree.fit(X_train, Y_train)



  #RandomForest

  from sklearn.ensemble import RandomForestClassifier

  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

  forest.fit(X_train, Y_train)



  #print accuracy

  print('[0]logistic regression trainning accuracy: ', log.score(X_train, Y_train))

  print('[1]knn trainning accuracy: ', knn.score(X_train, Y_train))

  print('[2]svc_lin trainning accuracy: ', svc_lin.score(X_train, Y_train))

  print('[3]gauss trainning accuracy: ', gauss.score(X_train, Y_train))

  print('[4]tree trainning accuracy: ', tree.score(X_train, Y_train))

  print('[5]forest trainning accuracy: ', forest.score(X_train, Y_train))

  return log, knn, svc_lin, gauss, tree, forest

model = models(X, Y)
## fill Missing Values

test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())

test.drop(['Alley'], axis = 1, inplace = True)

test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])

test['FireplaceQu'] = test['FireplaceQu'].fillna('NA')

test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])

test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])

test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])

test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])

test['GarageYrBlt'] = 2020 - test['GarageYrBlt']

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

test.drop(['PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)

test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['BsmtExposure'] = test['BsmtExposure'].fillna('NA')

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())

test['BsmtFinType1'] = test['BsmtFinType1'].fillna('NA')

test['BsmtFinType2'] = test['BsmtFinType2'].fillna('NA')

test['Electrical'] = test['Electrical'].fillna('Mix')

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())



list_col_object = []

for str in test.columns:

    if test[str].dtype == 'O':

        list_col_object.append(str)

print(list_col_object)





#encode the sex column

for str in list_col_object:

    print(str + ": ")

    print(test[str].unique())
list_cols_to_fill_object = []

list_cols_to_fill_numeric = []

for str in test.columns:

    if test[str].isnull().sum() != 0:

        if test[str].dtype == 'O':

            list_cols_to_fill_object.append(str)

            

        else:

            list_cols_to_fill_numeric.append(str)

print(list_cols_to_fill_numeric)

print(list_cols_to_fill_object)
for str in list_cols_to_fill_numeric:

    test[str] = test[str].fillna(test[str].mean())

for str in list_cols_to_fill_object:

    test[str] = test[str].fillna(test[str].mode()[0])

sns.heatmap(test.isnull(), cbar = False)



from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



#encode the sex column

for str in list_col_object:

    test[str] = labelencoder.fit_transform( test[str].values)

test.shape
import xgboost

classifier = xgboost.XGBRegressor()

regressor = xgboost.XGBRegressor()
booster = ['gbtree', 'gblinear']

base_score = [0.25, 0.5, 0.75, 1]
n_estimator = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

learning_rate= [0.05, 0.1, 0.15, 0.20]

min_child_weight = [1, 2, 3, 4]



hyperparameter_grid = {

    'n_estimator' : n_estimator,

    'max_depth' : max_depth,

    'learning_rate' : learning_rate,

    'min_child_weight' : min_child_weight,

    'booster' : booster,

    'base_score' : base_score

}
from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=regressor, param_distributions = hyperparameter_grid, cv = 5, n_iter =50, scoring = 'neg_mean_absolute_error', n_jobs = 4, verbose = 5, return_train_score = True, random_state = 42 )
random_cv.fit(X,Y)
random_cv.best_estimator_
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,

       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)
regressor.fit(X,Y)
import pickle

filename = 'finalized_model.pkl'

pickle.dump(classifier, open(filename, 'wb'))

X_test = test.iloc[:, 1:75].values

pred = regressor.predict(X_test)

print(pred)
submission=pd.DataFrame()

ID = test['Id']

submission['Id'] = ID

submission['SalePrice'] = pred
submission.to_csv('submission.csv', index = False)