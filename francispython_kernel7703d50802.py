# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/busara.csv')
data.head()
data['survey_date'] = pd.to_datetime(data['survey_date'])
data['month'] = data['survey_date'].dt.strftime('%m')
data['year'] = data['survey_date'].dt.strftime('%Y')
data['day'] = data['survey_date'].dt.strftime('%w')


data.head()
data['year'] = pd.get_dummies(data['year'])
data.head()
s  = ['survey_date']
data = data.drop(s,axis=True)
data = data.dropna(axis=1)
data.info()
x = data.drop(['depressed'], axis=1)
y = data.depressed
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
rf = LogisticRegression(C = 1000.0,random_state = 0)
rf.fit(X_train_std, y_train)
#rf.fit(x,y)
from sklearn.metrics import mean_absolute_error
pred_train = rf.predict(X_train_std)
print (mean_absolute_error(pred_train,y_train))
test = pd.read_csv('../input/test.csv')
test.head()
import seaborn as sns
sns.heatmap(test.isnull(),cbar=False, cmap='viridis')
test = test.dropna(axis=1)
test.head()
sns.heatmap(test.isnull(),cbar=False, cmap='viridis')
test['survey_date'] = pd.to_datetime(test['survey_date'])
test['month'] = test['survey_date'].dt.strftime('%m')
test['year'] = test['survey_date'].dt.strftime('%Y')
test['day'] = test['survey_date'].dt.strftime('%w')

test['year'] = pd.get_dummies(test['year'])
test = test.drop(s, axis=True)
test.info()
test.columns
q = ['fs_adskipm_often', 'asset_niceroof', 'cons_allfood',
       'cons_ed', 'med_vacc_newborns','ent_nonagbusiness',
       'cons_other','early_survey']
test = test.drop(q, axis=True)
x_test = test.drop(['surveyid'], axis=1)
test_pred = rf.predict(x_test)
from sklearn.model_selection import train_test_split
X_trains, X_tests, y_trains, y_tests = train_test_split(x_test,test_pred, test_size=0.33)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_trains)
X_train_st = sc.transform(X_trains)
X_test_st = sc.transform(X_tests)
rl = LogisticRegression(C = 1000.0,random_state = 0)
rl.fit(X_train_st, y_trains)
from sklearn.metrics import mean_absolute_error
pred_trains = rl.predict(X_train_st)
print (mean_absolute_error(pred_trains,y_trains))
q = {'surveyid': test["surveyid"], 'depressed': test_pred}
pred = pd.DataFrame(data=q)
pred = pred[['surveyid','depressed']]
pred.head
pred.to_csv('pred_set13.csv', index=False) #save to csv file
