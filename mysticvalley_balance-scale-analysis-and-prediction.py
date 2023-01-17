#Importing Libraries 

#basics and Visualization

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder



#ML libraries

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split





#metrics

from statistics import mean

from sklearn.metrics import accuracy_score as score

from sklearn.metrics import explained_variance_score as evs





#Ignore Warning 

import warnings as wrn

wrn.filterwarnings('ignore')
df = pd.read_csv(r'../input/balance-scale.csv')

df.head()
df.info()
#Visualization after doing label encoding

df['Class'] = LabelEncoder().fit_transform(df['Class'].tolist())

#pairplot

sns.pairplot(data=df)



#Heatmap

num_feat = df.select_dtypes(include=np.number).columns

plt.figure(figsize= (15, 15))

sns.heatmap(df.corr())

#Dividing X and y

y = df[['Class']]

X = df.drop(['Class'], axis = 1)



print(y.head())

print(X.head())
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 20)
#Classification and prediction

#ExtraTreeClassifier



clf = ExtraTreesClassifier(n_estimators=1000)

clf.fit(train_X, train_y)

pred = clf.predict(test_X)

print('Accuracy in percent = ',score(pred, test_y)*100)
#Classification and prediction

#XGBoost



clf = XGBClassifier(learning_rate=0.5, n_jobs=-1, n_estimators=1000)

clf.fit(train_X, train_y)

pred = clf.predict(test_X)

print('Accuracy in percent = ',score(pred, test_y)*100)
#Classification and prediction

#Random Forest



clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, )

clf.fit(train_X, train_y)

pred = clf.predict(test_X)

print('Accuracy in percent = ',score(pred, test_y)*100)
#Classification and prediction

#DT



clf = DecisionTreeClassifier()

clf.fit(train_X, train_y)

pred = clf.predict(test_X)

print('Accuracy in percent = ',score(pred, test_y)*100)
#Classification and prediction

#SVM



clf = SVC()

clf.fit(train_X, train_y)

pred = clf.predict(test_X)

print('Accuracy in percent = ',score(pred, test_y)*100)
#Classification and prediction

#KNN



clf = KNeighborsClassifier(n_neighbors=9)

clf.fit(train_X, train_y)

pred = clf.predict(test_X)

print('Accuracy in percent = ',score(pred, test_y)*100)