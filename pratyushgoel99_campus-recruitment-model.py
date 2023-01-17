# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data
data.info()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
sns.set(style='whitegrid', palette='muted', font_scale=1.1)
data_p = data[data['status'] == 'Placed']

data_np = data[data['status'] == 'Not Placed']

data2 = data.copy()

data2['status'] = data2['status'].map({'Placed':1, 'Not Placed': 0}).astype(int)
plt.figure(figsize=(14,7))

plt.title('Heatmap')

sns.heatmap(data=data2.drop(['salary', 'sl_no'], axis = 1).corr(), annot = True)
sns.pairplot(data, vars=['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'], hue='status', kind='reg')
data2[['gender','status']].groupby(['gender'], as_index=False).mean()
plt.figure(figsize=(12,6))

sns.swarmplot(x=data['status'], y=data['ssc_p'], hue=data['ssc_b'])
data2[['ssc_b','status']].groupby(['ssc_b'], as_index=False).mean()
data[['salary', 'ssc_b']].groupby('ssc_b', as_index=False).median()
plt.figure(figsize=(12,6))

sns.swarmplot(x=data['status'], y=data['hsc_p'], hue=data['hsc_b'])
data2[['hsc_b','status']].groupby(['hsc_b'], as_index=False).mean()
data[['salary', 'hsc_b']].groupby('hsc_b', as_index=False).median()
plt.figure(figsize=(12,6))

sns.swarmplot(x=data['status'], y=data['hsc_p'], hue=data['hsc_s'])
data2[['hsc_s','status']].groupby(['hsc_s'], as_index=False).mean()
data[['salary', 'hsc_s']].groupby('hsc_s', as_index=False).median()
plt.figure(figsize=(12,6))

sns.swarmplot(x=data['status'], y=data['degree_p'], hue=data['degree_t'])
data2[['degree_t','status']].groupby(['degree_t'], as_index=False).mean()
data[['salary', 'degree_t']].groupby('degree_t', as_index=False).median()
data2[['workex','status']].groupby(['workex'], as_index=False).mean()
plt.figure(figsize=(12,6))

sns.barplot(x='workex', y='status', data=data2)
data[['salary', 'workex']].groupby('workex', as_index=False).median()
plt.figure(figsize=(12,6))

sns.swarmplot(x=data['status'], y=data['etest_p'])
plt.figure(figsize=(12,6))

sns.regplot(x=data['salary'], y=data['etest_p'])
plt.figure(figsize=(12,6))

sns.swarmplot(x=data['status'], y=data['mba_p'], hue=data['specialisation'])
data2[['specialisation','status']].groupby(['specialisation'], as_index=False).mean()
data2[['specialisation','salary']].groupby(['specialisation'], as_index=False).median()

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv', index_col='sl_no')

data.drop('salary', axis=1, inplace=True)

data['status'] = data['status'].map({'Placed':1, 'Not Placed': 0}).astype(int)

data.head()
x = data.copy()

x.drop('status', axis=1, inplace = True)

y = data['status']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.16, random_state = 1)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
from sklearn.preprocessing import OneHotEncoder



cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

ohc = OneHotEncoder(handle_unknown='ignore', sparse=False)



n_cols_train = pd.DataFrame(ohc.fit_transform(x_train[cols]))

n_cols_test = pd.DataFrame(ohc.fit_transform(x_test[cols]))



n_cols_train.index = x_train.index

n_cols_test.index = x_test.index



n_cols_train.columns = ohc.get_feature_names(cols)

n_cols_test.columns = ohc.get_feature_names(cols)



x_train = pd.concat([x_train, n_cols_train], axis = 1)

x_test = pd.concat([x_test, n_cols_test], axis = 1)



x_train.drop(cols, axis = 1, inplace = True)

x_test.drop(cols, axis = 1, inplace = True)
x_train.head()
mean = x_train.mean()

std = x_train.std()
x_train = (x_train-mean)/std

x_test = (x_test-mean)/std

x_train.head(3)
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from xgboost import XGBClassifier

import time

from sklearn.feature_selection import RFE, f_regression

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from pprint import pprint

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
models = pd.DataFrame(columns=['model', 'score', 'std','Time to Train']) #DataFrame to store scores of all models



options = [GaussianNB(), 

           LogisticRegression(), 

           SVC(), 

           LinearSVC(), 

           DecisionTreeClassifier(), 

           RandomForestClassifier(), 

           KNeighborsClassifier(), 

           SGDClassifier(), 

           XGBClassifier()]   



model_names = ['Naive Bayes', 

               'Logistic Regression', 

               'Support Vector Machine', 

               'Linear SVC', 

               'Decison Tree',

               'Random Forest',

               'KNN', 

               'SGD Classifier',

               'XGBoost']  



for (opt, name) in zip(options, model_names):

    start=time.time()

    model = opt

    model.fit(x_train, y_train)

    

    scores = cross_val_score(model, x_train, y_train, cv = 5, scoring="accuracy")

    end=time.time()

    row = pd.DataFrame([[name, scores.mean(), scores.std(), end-start]], columns=['model', 'score', 'std','Time to Train'])

    models = pd.concat([models, row], ignore_index=True)



models.sort_values(by='score', ascending=False)
rf = RandomForestClassifier(random_state = 3, oob_score=True)

rf.fit(x_train, y_train)

print("OOB Score: ", rf.oob_score_)
model = RandomForestClassifier(random_state = 3)

model.fit(x_train, y_train)
rfe = RFE(model, n_features_to_select=1, verbose =3)

rfe.fit(x_train,y_train)



imp1 = pd.DataFrame({'feature':x_train.columns, 'rank1':rfe.ranking_})

imp1 = imp1.sort_values(by = 'rank1')

imp1
imp2= pd.DataFrame({'featur':x_train.columns, 'importance':np.round(model.feature_importances_, 3)})

imp2['rank2'] = imp2['importance'].rank(ascending=False, method='min')

imp2 = imp2.sort_values(by = 'importance', ascending=False)

imp2
# importances['rank']=importances2['rank'].values

# importances=importances.sort_values('rank')

# importances



imp = pd.concat([imp1, imp2], axis=1)

imp['rank'] = imp['rank1'] + imp['rank2']

imp = imp.sort_values(by = 'rank')

imp = imp.drop(['featur', 'importance', 'rank1', 'rank2'], axis=1)

imp
x_temp = x_train[imp.feature]
features = [i for i in range(22)]

results = []



for i in features:

    rf = RandomForestClassifier(n_jobs=-1, random_state=3)

    cols = x_temp.columns[:i+1]

    x_t = x_temp[cols]

    scores = cross_val_score(rf, x_t, y_train, cv = 5, scoring="accuracy")

    results.append(scores.mean())

    print(i, " : ", np.round(scores.mean(),3), np.round(scores.std(),3))
fig, ax = plt.subplots(figsize=(12,6))



ax.minorticks_on()



# Customize the major grid

ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')

# Customize the minor grid

ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')



sns.lineplot(y = results, x = features)
n_f = 8

to_keep = x_temp.columns[:n_f+1]

x_train_fimp = x_train[to_keep]

x_test_fimp = x_test[to_keep]

x_train_fimp.head()
rf = RandomForestClassifier(random_state=3, oob_score=True)

rf.fit(x_train_fimp, y_train)

rf.oob_score_
x_train_final = x_train_fimp

x_test_final = x_test_fimp
# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())
rfc=RandomForestClassifier(random_state=42)

param_grid = { 

    'n_estimators': [200, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8,'none']

}

pprint(param_grid)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

# Fit the random search model

CV_rfc.fit(x_train_final, y_train)
CV_rfc.best_params_
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='entropy')

rfc1.fit(x_train_final, y_train)
pred=rfc1.predict(x_test_final)

print("Accuracy for Random Forest after Hyperparameter Tuning on test data: ",accuracy_score(y_test,pred))

pred=rf.predict(x_test_final)

print("Accuracy for Random Forest before Hyperparameter Tuning on test data: ",accuracy_score(y_test,pred))