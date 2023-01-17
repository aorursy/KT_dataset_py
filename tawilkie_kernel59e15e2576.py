# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

import seaborn as sns
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



df_ohe = pd.get_dummies(df)



df_ohe.head(5)



X = df_ohe[df_ohe.columns[1:]].corr()['SalePrice']

mask = ((X>0.6)|(X<-0.5))&(X<1)

X = X[mask]

X



# # Read the data

# train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
Y_train = df_ohe.SalePrice

predictor_cols = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea','GarageCars','GarageArea','ExterQual_TA','KitchenQual_TA']



# Create training predictors data

X_train = df_ohe[predictor_cols]



svm_clf = Pipeline([

    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),

    ])



svm_clf.fit(X_train,Y_train)


test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()



test_df_ohe = pd.get_dummies(test_df)

test_df_ohe.head()

X_test = test_df_ohe[predictor_cols]

X_test = X_test.fillna(0)

Y_predict = svm_clf.predict(X_test)



print(Y_predict)
my_submission = pd.DataFrame({'Id': test_df_ohe.Id, 'SalePrice': Y_predict})



my_submission.to_csv('Wilkie_Timothy_submission.csv', index=False)