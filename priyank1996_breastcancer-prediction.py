# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/breastCancer.csv')

df.head()
df.info()
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)

df.info()
import seaborn as sns

import matplotlib.pyplot as plt
sns.barplot(x = 'diagnosis', y = 'radius_mean',data = df, palette='RdBu_r')
sns.barplot(x = 'diagnosis', y = 'area_mean',data = df, palette='RdBu_r')
sns.barplot(x = 'diagnosis', y = 'symmetry_worst',data = df, palette='RdBu_r')
df.corr()
f, ax = plt.subplots(figsize = (9,6))

sns.heatmap(df.corr(), ax = ax)
labels = np.array(df.columns[1:])

labels
grp_df = df.groupby('diagnosis', as_index = False)[labels].mean()

grp_df
grp_melt = grp_df.melt('diagnosis',value_vars=labels)

sns.barplot(x = 'variable', y = 'value', data = grp_melt.head(6), hue ='diagnosis')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis', axis = 1), df.diagnosis) 
from sklearn.linear_model import LogisticRegression

model_l = LogisticRegression()

model_l.fit(X_train,y_train)

model_l.score(X_test, y_test)
from sklearn.tree import DecisionTreeClassifier

model_t = DecisionTreeClassifier()

model_t.fit(X_train, y_train)

model_t.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators = 30)

model_rf.fit(X_train, y_train)

print('Score: {0:0.4f}'.format(model_rf.score(X_test, y_test)))
Model = ['Logistic Regression','Decision Tree','Random Forest']

Accuracy = [model_l.score(X_test, y_test), model_t.score(X_test, y_test), model_rf.score(X_test, y_test)]

for i in range(3):

    print('Score for %s is: %0.2f' %(Model[i], Accuracy[i]))