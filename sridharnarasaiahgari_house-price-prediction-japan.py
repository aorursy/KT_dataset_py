# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
train.head()
test.head()
train.shape, test.shape
train['MSZoning'] = train['MSZoning'].infer_objects()

train['Street'] = train['Street'].infer_objects()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()



numerical_col = []

cat_col = []

for x in train.columns:

    if train[x].dtype == 'object':

        train[x] = train[x].infer_objects()

        cat_col.append(x)

        MD=train[x].mode()

        train[x].replace('None', np.nan, inplace=True)

        train[x].fillna(MD)

        train[x] = labelencoder_X.fit_transform(train[x].astype(str))

    else:

        MD=train[x].mean()

        train[x].replace('None', np.nan, inplace=True)

        train[x].fillna(MD,  inplace=True)

        numerical_col.append(x)

        

print('CAT col \n', cat_col)

print('Numerical col\n')

print(numerical_col)
import statsmodels.formula.api as sm

X = train.iloc[:, :-1].values

y = train['SalePrice']

X.shape

#train_opt = np.append(arr = np.ones((1460, 76)).astype(int), values = X, axis = 1)

X_opt = X

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()



X = pd.DataFrame(X)

X.head()

X = X.drop(X.columns[[3,11,21,30,40,42,43,49,51,60,61,63,64,66,68,69,74,76,77]], axis = 1)

y = train['SalePrice']

X.shape

#train_opt = np.append(arr = np.ones((1460, 76)).astype(int), values = X, axis = 1)

X_opt = X

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

X = pd.DataFrame(X)

X = X.drop(X.columns[[2,9,11,13,18,29,32,35,36,41,42,52,53,58]], axis = 1)

y = train['SalePrice']

X.shape

#train_opt = np.append(arr = np.ones((1460, 76)).astype(int), values = X, axis = 1)

X_opt = X

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

X = pd.DataFrame(X)

X = X.drop(X.columns[[24]], axis = 1)

y = train['SalePrice']

X.shape

#train_opt = np.append(arr = np.ones((1460, 76)).astype(int), values = X, axis = 1)

X_opt = X

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()



numerical_col = []

cat_col = []

for x in test.columns:

    if test[x].dtype == 'object':

        test[x] = test[x].infer_objects()

        cat_col.append(x)

        MD=test[x].mode()

        test[x].replace('None', np.nan, inplace=True)

        test[x].fillna(MD)

        test[x] = labelencoder_X.fit_transform(test[x].astype(str))

    else:

        MD=test[x].mean()

        test[x].replace('None', np.nan, inplace=True)

        test[x].fillna(MD,  inplace=True)

        numerical_col.append(x)



        

print('CAT col \n', cat_col)

print('Numerical col\n')

print(numerical_col)
test.isnull().sum()

X_test = test.iloc[:, :].values

X_test = pd.DataFrame(X_test)

#test.head()

#X_test.head()



X_test2 = X_test.drop(X_test.columns[[2,10,13,15,20,33,36,39,41,48,50,65,67,75,34,3,11,21,30,40,42,43,49,51,60,61,63,64,66,68,69,74,76,77]], axis = 1)



X_test2.head()
X.head()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X, y)
preds=logreg.predict_proba(X_test2)[:,1]

preds
test = pd.read_csv("../input/test.csv")

#test_set=test.drop(['PassengerId','Name','Age','Ticket','Cabin','Embarked','Sex'], axis=1)

test['SalePrice']=logreg.predict(X_test2.fillna(0))

test[['Id','SalePrice']].to_csv("submission.csv",header=False, index=False)