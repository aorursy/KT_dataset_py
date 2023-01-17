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
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans
train = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
train.isnull().any().any()
test.isnull().any().any()
y_train = train['Satisfied']

X_train = train.drop(['custId','Satisfied'], axis=1)



train.loc[train['TotalCharges'] == " ", "TotalCharges"] = train["MonthlyCharges"]

train['TotalCharges'] = pd.to_numeric(train['TotalCharges'])



cols = ['tenure','MonthlyCharges','TotalCharges']





getdummies_cols = ['gender', 'SeniorCitizen', 'Married', 'Children',

       'TVConnection', 'Channel1', 'Channel2', 'Channel3', 'Channel4',

       'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices',

       'Subscription', 'PaymentMethod']



X_train = pd.get_dummies( X_train[getdummies_cols] )

X_train = pd.concat([X_train, train[cols]],axis=1)
X_test = test.drop(['custId'], axis=1)



test.loc[test['TotalCharges'] == " ", "TotalCharges"] = test["MonthlyCharges"]

test['TotalCharges'] = pd.to_numeric(test['TotalCharges'])



cols = ['tenure','MonthlyCharges','TotalCharges']



getdummies_cols2 = ['gender', 'SeniorCitizen']



getdummies_cols = ['gender', 'SeniorCitizen', 'Married', 'Children',

       'TVConnection', 'Channel1', 'Channel2', 'Channel3', 'Channel4',

       'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices',

       'Subscription', 'PaymentMethod']



X_test = pd.get_dummies( X_test[getdummies_cols] )

X_test = pd.concat([X_test, test[cols]],axis=1)
remove = ['SeniorCitizen', 'gender_Female', 'gender_Male', 

          'TVConnection_Cable','Channel2_No','Channel4_Yes',

          'Channel5_No','Channel6_No','HighSpeed_No','Subscription_Biannually','tenure']



X_train = X_train.drop(remove,axis=1)

X_test = X_test.drop(remove,axis=1)
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(X_train)

X_train = transformer.transform(X_train)



transformer2 = RobustScaler().fit(X_test)

X_test = transformer2.transform(X_test)
X_train
from sklearn.cluster import KMeans,Birch



algo = KMeans(n_clusters=2,init='k-means++',max_iter=10000,n_init=10000,random_state=20).fit(X_test)



#algo = Birch(branching_factor=100, n_clusters=2, threshold=0.5, compute_labels=True).fit(X_test)
pred = algo.predict(X_test)
result = pd.DataFrame({'Satisfied':pred}, index=test['custId'])
pd.Series(pred).value_counts()
result.head()
submission = result.to_csv("submission.csv")