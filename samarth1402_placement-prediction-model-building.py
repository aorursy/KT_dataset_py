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



#SK-Learn        

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)



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



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv', index_col='sl_no')

data.drop('salary', axis=1, inplace=True)

data['status'] = data['status'].map({'Placed':1, 'Not Placed': 0}).astype(int)

data.head()
data.info()
y = data['status']

X = data.copy()

X.drop('status', axis=1, inplace=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16, random_state=1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import OneHotEncoder



cols = [ 'gender','ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

ohc = OneHotEncoder(handle_unknown='ignore', sparse=False)



n_cols_train = pd.DataFrame(ohc.fit_transform(X_train[cols]))

n_cols_test = pd.DataFrame(ohc.transform(X_test[cols]))



n_cols_train.index = X_train.index

n_cols_test.index = X_test.index



n_cols_train.columns = ohc.get_feature_names(cols)

n_cols_test.columns = ohc.get_feature_names(cols)



X_train = pd.concat([X_train, n_cols_train], axis=1)

X_test = pd.concat([X_test, n_cols_test], axis=1)



X_train.drop(cols, axis=1, inplace=True)

X_test.drop(cols, axis=1, inplace=True)
X_train.head()
mean = X_train.mean()

std = X_train.std()
X_train = (X_train-mean)/std

X_test = (X_test-mean)/std
X_train.head(3)




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

    model.fit(X_train, y_train)

    

    scores = cross_val_score(model, X_train, y_train, cv = 5, scoring="accuracy")

    end=time.time()

    row = pd.DataFrame([[name, scores.mean(), scores.std(), end-start]], columns=['model', 'score', 'std','Time to Train'])

    models = pd.concat([models, row], ignore_index=True)



models.sort_values(by='score', ascending=False)
rf = RandomForestClassifier(random_state = 3, oob_score=True)

rf.fit(X_train, y_train)

print("OOB Score: ", rf.oob_score_)
model = RandomForestClassifier(random_state = 3)

model.fit(X_train, y_train)
rfe = RFE(model, n_features_to_select=1, verbose =3)

rfe.fit(X_train,y_train)



imp1 = pd.DataFrame({'feature':X_train.columns, 'rank1':rfe.ranking_})

imp1 = imp1.sort_values(by = 'rank1')

imp1
imp2= pd.DataFrame({'featur':X_train.columns, 'importance':np.round(model.feature_importances_, 3)})

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
X_temp = X_train[imp.feature]
features = [i for i in range(22)]

results = []



for i in features:

    rf = RandomForestClassifier(n_jobs=-1, random_state=3)

    cols = X_temp.columns[:i+1]

    X_t = X_temp[cols]

    scores = cross_val_score(rf, X_t, y_train, cv = 5, scoring="accuracy")

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

to_keep = X_temp.columns[:n_f+1]

X_train_fimp = X_train[to_keep]

X_test_fimp = X_test[to_keep]

X_train_fimp.head()
rf = RandomForestClassifier(random_state=3, oob_score=True)

rf.fit(X_train_fimp, y_train)

rf.oob_score_
X_train_final = X_train_fimp

X_test_final = X_test_fimp
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

CV_rfc.fit(X_train_final, y_train)

CV_rfc.best_params_
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=7, criterion='entropy')

rfc1.fit(X_train_final, y_train)

pred=rfc1.predict(X_test_final)

print("Accuracy for Random Forest after Hyperparameter Tuning on test data: ",accuracy_score(y_test,pred))

pred=rf.predict(X_test_final)

print("Accuracy for Random Forest before Hyperparameter Tuning on test data: ",accuracy_score(y_test,pred))