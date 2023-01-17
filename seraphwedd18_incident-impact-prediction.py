import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt



train = pd.read_csv("../input/incident-impact-prediction/train.csv")

test = pd.read_csv("../input/incident-impact-prediction/test.csv")
Y = train.impact

train.drop(['impact'], axis=1, inplace=True)

X = train

del train
X.drop(['Unnamed: 0', 'created_at', 'updated_at'], axis=1, inplace=True)

print(test.columns)

test.drop(['S.No', 'created_at', 'updated_at'], axis=1, inplace=True)

print(Y.shape, X.shape)

print(Y.describe())

print(X.describe())

categorical = [col for col in X.columns if X[col].dtype==object]

numerical = [col for col in X.columns if X[col].dtype!=object]

print(categorical, len(categorical))

print(numerical, len(numerical))
for col in categorical:

    temp = {}

    count = 0

    for val in X[col].values:

        try:

            temp[val]

        except:

            temp[val] = count

            count += 1

    for val in test[col].values:

        try:

            temp[val]

        except:

            temp[val] = count

            count += 1

    X[col] = [temp[x] for x in X[col].values]

    test[col] = [temp[x] for x in test[col].values]

print(X[categorical])
#Eliminate the NAN

for col in X.columns:

    X.loc[X[col] == '?', col] = 0



#Check the unique values

import seaborn as sns

nu = X.nunique().reset_index()

nu.columns = ['features', 'uniques']

ax = sns.barplot(x='features', y='uniques', data=nu)

ax.tick_params(axis='x', rotation=90)

print(nu)
import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(X.corr(), annot=True, linewidth=0.02)

fig=plt.gcf()

fig.set_size_inches(20,20)

plt.show()
corr = X.corr()

drop_cols = []

for col in X.columns:

    if sum(corr[col].map(lambda x: abs(x) > 0.1)) <= 4:

        drop_cols.append(col)

X.drop(drop_cols, axis=1, inplace=True)

print(drop_cols)

display(X)
from sklearn import tree, feature_extraction

from sklearn.model_selection import cross_val_score

clf = tree.DecisionTreeClassifier(

    criterion='entropy',

    max_depth=100,

    random_state=11

)



print(cross_val_score(clf, X, Y, cv=5))
clf.fit(X, Y)

p = clf.predict(X)

predict = clf.predict(test.drop(drop_cols, axis=1))
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns



cfm = confusion_matrix(Y, p)

sns.heatmap(cfm, annot=True)

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()
#Save Prediciton

ss = pd.DataFrame(zip([x for x in range(1, len(predict)+1)], predict), columns=['ID', 'prediction1'])

print(ss.shape)

ss.to_csv('submission.csv', index=False)