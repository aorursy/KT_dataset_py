# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/pokemon.csv')
df2 = df.select_dtypes(include=['float64','int64'])
df3 = df.select_dtypes(include=['object'])
df2.info()
df3.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
labels = le.fit_transform(df['type1'])
print(len(le.classes_))
print(le.classes_)
type2_le = preprocessing.LabelEncoder()
type2 = type2_le.fit_transform(df['type2'].astype(str))
len(type2_le.classes_)
for i in df2:
    if df[i].isnull().values.any():
            df[i].fillna(df[i].mean(), inplace=True)
df[list(df2)].isnull().values.any()
data = {
    'attack': df['attack'],
    'defense': df['defense'],
    'sp_attack': df['sp_attack'],
    'sp_defense': df['sp_defense'],
    'type2': type2,
    'type1': df['type1']
}
data = pd.DataFrame(data)
data = df.filter(like='against').join(data)

X = data.drop('type1', axis=1)
y = data['type1']
print(list(X))
X.head()
from pandas.tools.plotting import radviz

plt.subplots(figsize=(20,15))
radviz(data, 'type1')
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=10, random_state=48)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

result = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

print(result.mean())
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=list(X),
                                class_names=le.classes_,
                                filled=True, rounded=True,
                                special_characters=True
                               ) 
graph = graphviz.Source(dot_data) 
graph
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=10, random_state=48)

clf = LogisticRegression()
result = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

print(result.mean())
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
clf = LogisticRegression()
clf.fit(X_train, y_train)
pd.DataFrame(clf.coef_ ,index=le.classes_, columns=list(X))
pd.DataFrame(clf.intercept_, index=le.classes_)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
plt.subplots(figsize=(20,15))
sns.heatmap(cm, annot=True)
