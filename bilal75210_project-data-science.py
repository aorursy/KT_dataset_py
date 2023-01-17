# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn import tree

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_predict

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/FIFA 2018 Statistics.csv')

df.head()
df.info()
#dummy=pd.get_dummies(df.loc[:,'Man of the Match'])

#df=df.merge(dummy,left_index=True,right_index=True)

df.head()
df.corr()
df.describe()
df.isnull().sum()
for col in df.columns:

    col.rstrip()
tree = DecisionTreeClassifier()

df.drop(['Own goal Time', 'Own goals', '1st Goal'], axis = 1, inplace= True)

df.drop(['Corners', 'Fouls Committed', 'On-Target'], axis = 1, inplace=True)

df.drop('Date', axis = 1, inplace=True)

one_hot_data = pd.get_dummies(df,drop_first=True)

#one_hot_data.head()

one_hot_data.info()
#one_hot_data.dropna(inplace=True)

#tree.fit(one_hot_data, one_hot_data.loc[:,'PSO_Yes'])
one_hot_data.info()
def cal_accuracy(y_test, y_pred): 

      

    print("Confusion Matrix: ", 

        confusion_matrix(y_test, y_pred)) 

      

    print ("Accuracy : ", 

    accuracy_score(y_test,y_pred)*100) 

      

    print("Report : ", 

    classification_report(y_test, y_pred))
#idx = pd.IndexSlice

#one_hot_data.fillna(one_hot_data['Own goal Time'].mean(),inplace=True)

#one_hot_data.fillna(value=0,inplace=True)
one_hot_data.isnull().values.any()
one_hot_data.head()
df = one_hot_data.copy()

df.describe()
df = df.apply(LabelEncoder().fit_transform)

df.head()
label = df['Man of the Match_Yes']



features = df.drop(['Man of the Match_Yes'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state =0) 
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 

clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test) 

print("Predicted values:") 

print(y_pred) 

    

cal_accuracy(y_test, y_pred)
y_pred = clf_gini.predict(X_train) 

print("Predicted values:") 

print(y_pred) 

    

cal_accuracy(y_train, y_pred)
scores = cross_val_score(clf_gini, features, label, cv=5)

scores  
one_hot_data.head()
model = GaussianNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cal_accuracy(y_test, y_pred)
y_pred = model.predict(X_train)

cal_accuracy(y_train, y_pred)
scores = cross_val_score(model, features, label, cv=5)

scores  
clf = RandomForestClassifier(n_estimators=100, max_depth=1,random_state=0)

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

cal_accuracy(y_test, y_pred)
y_pred=clf.predict(X_train)

cal_accuracy(y_train, y_pred)
scores = cross_val_score(clf, features, label, cv=5)

scores  
models = pd.DataFrame({

        'Model'          : ['Naive Bayes',  'Decision Tree', 'Random Forest'],

        'Training_Score' : [model.score(X_train,y_train),  clf_gini.score(X_train,y_train), clf.score(X_train,y_train)],

        'Testing_Score'  : [model.score(X_test,y_test), clf_gini.score(X_test,y_test), clf.score(X_test,y_test)]

    })

models.sort_values(by='Testing_Score', ascending=False)