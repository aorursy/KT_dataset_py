import pandas as pd

import numpy as np
df = pd.read_csv('../input/adult.csv', header = 0)

df.head()
df.rename(columns={'native.country':'nativecountry'}, inplace=True)
df = df[(df.workclass != '?')]
df = df[(df.nativecountry != '?')]
df = df[(df.occupation != '?')]
df = df.drop('education',1)
df.occupation.unique()
d = {'Private' : 1, 'Self-emp-not-inc' : 2, 'Self-emp-inc' : 3, 'Federal-gov' : 4, 'Local-gov' : 5, 

     'State-gov' : 6, 'Without-pay' : 7, 'Never-worked' : 8}

df['workclass'] = df['workclass'].map(d)

d = {'Married-civ-spouse' : 1, 'Divorced' : 2, 'Never-married' : 3, 'Separated' : 4,

     'Widowed' : 5, 'Married-spouse-absent' : 6, 'Married-AF-spouse' : 7}

df['marital.status'] = df['marital.status'].map(d)

d = {'Tech-support' : 1, 'Craft-repair' : 2, 'Other-service' : 3, 'Sales' : 4, 'Exec-managerial' : 5,

     'Prof-specialty' : 6, 'Handlers-cleaners' : 7, 'Machine-op-inspct' : 8, 'Adm-clerical' : 9,

     'Farming-fishing' : 10, 'Transport-moving' : 11, 'Priv-house-serv' : 12, 'Protective-serv' : 13, 'Armed-Forces' : 14}

df['occupation'] = df['occupation'].map(d)

d = {'Wife' : 1, 'Own-child' : 2, 'Husband' : 3, 'Not-in-family' : 4, 'Other-relative' : 5, 'Unmarried' : 7}

df['relationship'] = df['relationship'].map(d)

d = {'White' : 1, 'Asian-Pac-Islander' : 2, 'Amer-Indian-Eskimo' : 3, 'Other' : 4, 'Black' : 5}

df['race'] =df['race'].map(d)

d = {'Female' : 1, 'Male' : 2}

df['sex'] = df['sex'].map(d)

d = {'United-States' : 1, 'Mexico' : 2, 'Greece' : 3, 'Vietnam' : 4, 'China' : 5, 'Taiwan' : 6,

       'Holand-Netherlands' : 7, 'Puerto-Rico' : 8, 'Poland' : 9, 'Iran' : 10, 'England' : 11,

       'Germany' : 12, 'Italy' : 13, 'Japan' : 14, 'Hong' : 15, 'Honduras' : 16, 'Cuba' : 17, 'Ireland' : 18,

       'Cambodia' : 19, 'Peru' : 20, 'Nicaragua' : 21, 'Dominican-Republic' : 22, 'Haiti' : 23,

       'Hungary' : 24, 'Columbia' : 25, 'Guatemala' : 26, 'El-Salvador' : 27, 'Jamaica' : 28,

       'Ecuador' : 29, 'France' : 30, 'Yugoslavia' : 31, 'Portugal' : 32, 'Laos' : 33, 'Thailand' : 34,

       'Outlying-US(Guam-USVI-etc)' : 35, 'Scotland' : 36,

       'India' : 35, 'Philippines' : 36, 'Trinadad&Tobago' : 37, 'Canada' : 38, 'South' : 39}

df['nativecountry'] = df['nativecountry'].map(d)

d = {'>50K' : 1, '<=50K' : 2}

df['income'] = df['income'].map(d)
from sklearn import tree
features = list(df.columns[:13])

features
y = df["income"]

x = df[features]

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)
output = Tree.predict([40,3,77058,12,1,4,3,1,2,0,4350,20,1])

print (output)
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

Forest = RandomForestClassifier(n_estimators = 200)

Forest = Forest.fit(x,y)

output = Forest.predict([40,3,77058,12,1,4,3,1,2,0,4350,20,1])

print(output)
from sklearn.cross_validation import train_test_split

from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=0)
kf = KFold(len(df), n_folds=10, shuffle=False)

print('KFold CrossValScore Using Random Forest %s' % cross_val_score(Forest,x, y, cv=5).mean())
kf = KFold(len(df), n_folds=10, shuffle=False)

print('KFold CrossValScore Using Decision Tree %s' % cross_val_score(Tree,x, y, cv=5).mean())
rf = Forest.fit(X_train, y_train)

y_pred = rf.predict(X_test)

metrics.accuracy_score(y_test, y_pred)