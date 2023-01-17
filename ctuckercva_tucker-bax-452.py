# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

testdf = pd.read_csv('../input/test.csv')
print(testdf.head())

testdf.columns.values

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

testdf = testdf.drop(['Name', 'Ticket', 'Cabin'], axis=1)
print(df.isnull().sum())

print(testdf.isnull().sum())
df['Embarked'].fillna('S', inplace = True)

df['Age'].fillna(df['Age'].median(), inplace = True)

testdf['Embarked'].fillna('S', inplace = True)

testdf['Age'].fillna(df['Age'].median(), inplace = True)

testdf['Fare'].fillna(df['Fare'].median(), inplace = True)
df.groupby('Survived').count()
print(df.isnull().sum())

print(testdf.isnull().sum())

df.dtypes.sample(9)
#change object variables into numerical

df2 = pd.get_dummies(df)

df2.drop(['Survived', 'PassengerId'], axis = 1, inplace = True)

print(df2.columns.values)

testdf2 = pd.get_dummies(testdf)

print(testdf2.columns.values)

#OK now we have a workable data set

from sklearn import tree

import graphviz

clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf.fit(df2, df['Survived'])
import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None, feature_names = df2.columns.values) 

graph = graphviz.Source(dot_data)

graph
predictions = clf.predict(testdf2.drop('PassengerId', axis = 1))
output = pd.DataFrame({'PassengerId':testdf2['PassengerId'], 'Survived': predictions})

print(output)
output.to_csv('../working/Tucker BAX 452.csv', index = False)
pwd