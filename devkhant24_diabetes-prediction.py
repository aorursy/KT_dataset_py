# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import xgboost

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
corrmat = df.corr()

top = corrmat.index

plt.figure(figsize=(10,10))

sns.heatmap(df[top].corr(),annot=True,cmap='RdYlGn')
x = df.drop('Outcome',axis=1)

y = df['Outcome']
os = RandomOverSampler(ratio=1)

xos,yos = os.fit_sample(x,y)
xg = xgboost.XGBClassifier(criterion='entropy')

dt = DecisionTreeClassifier(random_state=1,criterion='entropy')

rf = RandomForestClassifier(random_state=1,criterion='entropy')

lr = LogisticRegression()
score = cross_val_score(rf,xos,yos,cv=5)

score.mean()
a = [xg,dt,lr,rf]

acc = []

for i in a:

    i.fit(xos,yos)

    pred = i.predict(xos)

    acc.append(accuracy_score(pred,yos))

print(acc)
rf.fit(xos,yos)
pred = rf.predict(xos)
accuracy_score(pred,yos)
confusion_matrix(yos,pred)
print(classification_report(yos,pred))