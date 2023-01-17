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
import pandas as pd
import matplotlib.pyplot as plt 
sonar=pd.read_csv("/kaggle/input/sonardata/sonar.csv")
sonar.head(5)
from sklearn.model_selection import train_test_split
X=sonar.iloc[:,0:60]
y=sonar.iloc[:,60]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_train, y_train)
score=rnd_clf.feature_importances_
fname=sonar.columns[:60]
df = pd.Series(score,index=fname)
df[0:5]
df.plot.barh(x='Method', y='Accuracy',figsize=(15,12))
df.sort_values(ascending=False, inplace=True)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))
nc=np.arange(5,60,5)
acc=np.empty(11)
i=0
for k in np.nditer(nc):
    topf=df.index[1:k]
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train[topf], y_train)
    y_pred_tree = tree_clf.predict(X_test[topf])
    acc[i]=accuracy_score(y_test, y_pred_tree)
    i = i + 1
acc
x=pd.Series(acc,index=nc)
x.plot()
# Add title and axis names
plt.title('Top Features versus accuracy')
plt.xlabel('Numer of Fetures')
plt.ylabel('Accuracy')
plt.show()
top10=df.index[1:30]
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train[top10], y_train)
y_pred_rf = rnd_clf.predict(X_test[top10])
print(accuracy_score(y_test, y_pred_rf))
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
y_pred_ad = ada_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_ad))
dfig = pd.DataFrame()
nc=np.arange(0.1,0.7,0.05)
 
for k in np.nditer(nc):
    ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=k, random_state=42)
    ada_clf.fit(X_train, y_train)
    y_pred_tree = ada_clf.predict(X_test)
    a_row = pd.Series([k, accuracy_score(y_test, y_pred_tree)])
    row_df = pd.DataFrame([a_row])
    dfig = pd.concat([row_df, dfig], ignore_index=False)

dfig.columns=['Learning Rate','Accuracy']
dfig.sort_values(ascending=True, inplace=True,by=['Learning Rate'])
dfig.plot.line(x='Learning Rate', y='Accuracy',figsize=(15,8))
dfig.head(8)
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, max_depth=2, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_gb))
dfig = pd.DataFrame()
nc=np.arange(2,10,1)
 
for k in np.nditer(nc):
    gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, max_depth=k, random_state=42)
    gb_clf.fit(X_train, y_train)
    y_pred_tree = gb_clf.predict(X_test)
    a_row = pd.Series([k, accuracy_score(y_test, y_pred_tree)])
    row_df = pd.DataFrame([a_row])
    dfig = pd.concat([row_df, dfig], ignore_index=False)
dfig.columns=['Max Depth','Accuracy']
dfig.sort_values(ascending=True, inplace=True,by=['Max Depth'])
dfig.plot.line(x='Max Depth', y='Accuracy',figsize=(15,8))
dfig.head(10)
dfig = pd.DataFrame()
nc=np.arange(20,0,-2)
 
for k in np.nditer(nc):
    gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, max_depth=3,min_samples_leaf=int(k), random_state=42)
    gb_clf.fit(X_train, y_train)
    y_pred_tree = gb_clf.predict(X_test)
    a_row = pd.Series([k, accuracy_score(y_test, y_pred_tree)])
    row_df = pd.DataFrame([a_row])
    dfig = pd.concat([row_df, dfig], ignore_index=False)

dfig.columns=['Min Samples Leaves','Accuracy']
dfig.sort_values(ascending=True, inplace=True,by=['Min Samples Leaves'])
dfig.plot.line(x='Min Samples Leaves', y='Accuracy',figsize=(15,8))
dfig = pd.DataFrame()
nc=np.arange(10,300,20)
 
for k in np.nditer(nc):
    gb_clf = GradientBoostingClassifier(n_estimators=k, learning_rate=0.5, max_depth=3,min_samples_leaf=1, random_state=42)
    gb_clf.fit(X_train, y_train)
    y_pred_tree = gb_clf.predict(X_test)
    a_row = pd.Series([k, accuracy_score(y_test, y_pred_tree)])
    row_df = pd.DataFrame([a_row])
    dfig = pd.concat([row_df, dfig], ignore_index=False)
dfig.columns=['Number of Stages','Accuracy']
dfig.sort_values(ascending=True, inplace=True,by=['Number of Stages'])
dfig.plot.line(x='Number of Stages', y='Accuracy',figsize=(15,8))
dfig.head(15)