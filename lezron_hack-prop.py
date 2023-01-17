# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn import preprocessing

from sklearn import utils

from sklearn import metrics,svm

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

li = df['Date_Time'][0:5]

for f in df:

    for i in f:

        if(i==-200):

            df[f][i] = 0



df['sec'] = df.Date_Time.str.slice(-8,-6).astype(int)

df['mounth'] = df.Date_Time.str.slice(3,5).astype(int)

df['date'] = df.Date_Time.str.slice(0,2).astype(int)

df.columns
X = df[['AH', 'C6H6(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',

       'PT08.S4(NO2)', 'PT08.S5(O3)', 'RH']]

X = preprocessing.normalize(X,norm='l1')

np.std(X)
y = df[['T']]

mean = np.mean(y)

y = preprocessing.normalize(y,norm='l1')

y = y/100

np.std(y)

#y[0:10]

mean
sns.heatmap(df.corr())
clf = svm.SVR()

lab_enc = preprocessing.LabelEncoder()

y_Enc = lab_enc.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_Enc, test_size=0.3)
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

y_predict[0:10]
print("Accuracy:",metrics.mean_squared_error(y_test, y_predict))
test = pd.read_csv('../input/test.csv')

test['sec'] = test.Date_Time.str.slice(-8,-6).astype(int)

test['mounth'] = test.Date_Time.str.slice(3,5).astype(int)

test['date'] = test.Date_Time.str.slice(0,2).astype(int)
X_testing = test[['AH', 'C6H6(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',

       'PT08.S4(NO2)', 'PT08.S5(O3)', 'RH']]
y_pred = clf.predict(X_testing)
test.columns
submission = pd.DataFrame({

    'Date_Time':test['Date_Time'],

    'T':y_pred

})
submission.to_csv("submit.csv", index = False)