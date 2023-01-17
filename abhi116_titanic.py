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
import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")

df.head()
df.describe(include='all')
df.isna().sum()
plt.figure(figsize=(20, 10))

df['Age'].plot(kind='hist', bins=25)
df.describe()
dummy1 = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)

dummy2 = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)



df = pd.concat([df, dummy1, dummy2], axis=1)

df.head()
import seaborn as sns



corr = df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(240, 15, n=9),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
X = df[['Pclass', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

y = df['Survived']

X.head()
X['Fare'] = X['Fare']/X['Fare'].max()

X.head()
from sklearn.model_selection import train_test_split

train, test, train_labels, test_labels = train_test_split(X, y, test_size=1/3)
from sklearn.svm import SVC

model = SVC(C=2500, gamma=0.003)

model.fit(X, y)
accuracy = model.score(X, y)

print('Accuracy = ' , accuracy)
test_df = pd.read_csv("../input/test.csv")

test_df.head()
d1 = pd.get_dummies(test_df['Sex'], prefix='Sex', drop_first=True)

d2 = pd.get_dummies(test_df['Embarked'], prefix='Embarked', drop_first=True)



test_df = pd.concat([test_df, d1, d2], axis=1)

test_df.head()

test_df['Fare'] = test_df['Fare']/test_df['Fare'].max()
test_set = test_df[['Pclass', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

test_set.isnull().sum()
test_set['Fare'] = test_set['Fare'].interpolate()
result = model.predict(test_set)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],

                             'Survived': result})

submission.head()
submission.to_csv('Submission.csv', index=False)