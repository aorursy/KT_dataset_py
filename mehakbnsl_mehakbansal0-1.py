# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import pickle

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve, cross_val_score, cross_val_predict



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

sys.path.append("/kaggle/input/enron-project/")

from feature_format import featureFormat

from feature_format import targetFeatureSplit





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#sys.path.append('/kaggle/input/enron-project/')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
with open("/kaggle/input/enron-project/final_project_dataset_unix.pkl", "rb") as file:

    enron_dict = pickle.load(file)



df = pd.DataFrame.from_dict(enron_dict,orient='index')



df = df.replace('NaN',np.nan)

df.head()



plt.scatter(df.salary,df.bonus)

plt.xlabel('salary')

plt.ylabel('bonus')
x=df.loc[:,'salary']

x=x.to_dict()

key=max(x,key=x.get)

print(key)

x=enron_dict.pop('TOTAL')

df=pd.DataFrame.from_dict(enron_dict,orient='index')

df=df.replace('NaN',np.nan)

plt.scatter(df.salary,df.bonus)

plt.xlabel('salary')

plt.ylabel('bonus')

plt.scatter(df.total_payments,df.salary)

plt.xlabel('Total_payments')

plt.ylabel('salary')

plt.show()

plt.scatter(df.shared_receipt_with_poi,df.to_messages)

plt.xlabel('shared_receipt_with_poi')

plt.ylabel('to_messages')

guess = df[(df.salary > 1000000) | (df.bonus > 5000000)|(df.total_payments.astype(float)>100000000)][['salary','bonus','total_payments','shared_receipt_with_poi','poi']]



guess
list=['salary','to_messages','deferral_payments','total_payments','loan_advances','bonus','restricted_stock_deferred','deferred_income','total_stock_value','expenses','from_poi_to_this_person','exercised_stock_options','from_messages','other','from_this_person_to_poi','poi','long_term_incentive','shared_receipt_with_poi','restricted_stock','director_fees']
#plotting the heatmap

plt.figure(figsize=(5,5))

sns.heatmap(df.dropna(how='all').drop(['other'],axis=1).corr(),cmap='coolwarm')

plt.show()

sns.set(rc={"font.style":"normal",

            'axes.labelsize':15,

            'xtick.labelsize':10,

            'font.size':8,

            'ytick.labelsize':10}

       )

financial=['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock' ]

sns.pairplot(df[financial],hue='poi',palette='husl')
sns.set(rc={'axes.labelsize':8,

            'xtick.labelsize':8,

            'font.size':8,

            'ytick.labelsize':8}

       )

#financial=[ 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees' ]

email = ['to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'poi']

sns.pairplot(df[email], hue = 'poi', palette = 'husl')



plt.show()
features_list=['poi','salary','bonus','deferred_income','director_fees', 'to_messages','from_poi_to_this_person', 

                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','total_payments','restricted_stock_deferred', 'deferral_payments','total_stock_value', 'expenses', 'exercised_stock_options' ]

data = featureFormat(enron_dict,features_list)

y, X = targetFeatureSplit(data)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(len(y_train), len(y_test))

X_scaled = preprocessing.scale(X_train)

Xtest_scaled = preprocessing.scale(X_test)

lr = LogisticRegression(max_iter = 5000)

grid = {'C':[0.01, 0.03, 0.1, 0.3, 1, 3,10]}

grid1 = GridSearchCV(lr,param_grid=grid,scoring='accuracy',cv=5)

grid1.fit(X_scaled,y_train)
print(grid1.best_params_)

pred = grid1.predict(Xtest_scaled)

print('Test Accuracy = ',grid1.score(Xtest_scaled,y_test))

print(metrics.classification_report(y_test,pred, zero_division=0))
rf = RandomForestClassifier(n_estimators=200)

grid = {'n_estimators':[1, 10, 50],'max_depth':[25,30,35,40,45,50]}

grid_rf = GridSearchCV(rf,param_grid=grid,scoring='accuracy',cv=5)

grid_rf.fit(X_train,y_train)
print(grid_rf.best_params_)

pred = grid_rf.predict(X_test)

print('Accuracy = ',grid_rf.score(X_test,y_test))

print(metrics.classification_report(y_test,pred, zero_division = 0))
km = KNeighborsClassifier()

grid = {'n_neighbors':[4,5,6,7,8,9,10,11]}

grid_km = GridSearchCV(km,param_grid=grid,scoring='accuracy',cv=5)

grid_km.fit(X_train,y_train)
print(grid_km.best_params_)

pred = grid_km.predict(X_test)

print('Accuracy = ',grid_km.score(X_test,y_test))

print(metrics.classification_report(y_test,pred, zero_division = 0))
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],"base_estimator__splitter":["best", "random"], "n_estimators": [1, 2]}

grid_ada = GridSearchCV(ada, param_grid=param_grid, scoring = 'accuracy', cv=5)

grid_ada.fit(X_train, y_train)
print(grid_ada.best_estimator_)

pred = grid_ada.predict(X_test)

print('Accuracy = ',grid_ada.score(X_test,y_test))

print(metrics.classification_report(y_test,pred, zero_division = 0))
pickle.dump(ada, open("my_classifier.pkl", "wb") )

pickle.dump(enron_dict, open("my_dataset.pkl", "wb") )

pickle.dump(features_list, open("my_feature_list.pkl", "wb") )