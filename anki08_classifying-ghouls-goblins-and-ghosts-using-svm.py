# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import LabelEncoder

# data processing, CSV file I/O (e.g. pd.read_csv)





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
print(train.head())

print(train.info())

print(train.describe())



#this tels us that there is no missing value in any of the column

#the data lies between 0 and 1 there is no skewness in data

#color and type has to be changed to numerical type
train2=train

x=train.groupby('type')['color'].agg('count')

print(x)

y_train=train['type']

x_train=train.drop(['color','id','type'],axis=1)

le = LabelEncoder().fit(train2['color'])

color_  = le.transform(train2['color'])

le = LabelEncoder().fit(y_train)

y_train = le.transform(y_train)

#use labelEncoder() to transform data to numerical form



sns.pointplot(x=y_train,y=color_)  

plt.show()

#the plot shows that colour is not significant in predicting

#so we drop colour
#model



id=test['id']

#use gridsearch to find the best parameters

params = {'C':[1,5,10,0.1,0.01],'gamma':[0.001,0.01,0.05,0.5,1]}

svc = SVC()

#params={'min_samples_leaf':[40]}

clf = GridSearchCV(svc ,params, refit='True', n_jobs=1, cv=5)





clf.fit(x_train, y_train)

print(test.head())

x_test=test.drop(['id','color'],axis=1)

y_test = clf.predict(x_test)

print(clf.score(x_train,y_train))

#print(y_test[:])

#transform predicted data back

y_test2=le.inverse_transform(y_test)

#print((clf.score(x_train,y_train)))

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))

submission = pd.DataFrame( { 

                  "type": y_test2

                   },index=id)

print(submission.head())

submission.to_csv('submission.csv')
