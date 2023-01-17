# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'../input/train.csv')
test = pd.read_csv(r'../input/test.csv')
Y_Train = train['Survived']
train = train.drop(columns=['Survived'])

def setup(df):
    X = df.copy(deep=True)
    X['Cabin Letter'] = X['Cabin'].apply(lambda x: ord(x[:1]) if type(x) == str else x)
    X['Embarked'] = X['Embarked'].apply(lambda x: ord(x) if type(x) == str else x)
    X = pd.concat([X, pd.get_dummies(X['Sex'])], axis=1)
    X = X.drop(columns=['Name', 'Ticket', 'Cabin', 'Sex'])
    X = X.astype('float')
    X = X.fillna(X.mean())
    return X

X_Train = setup(train)
X_Test = setup(test)
best = SelectKBest(chi2, k=5).fit(X_Train, Y_Train)
sup = best.get_support()
used = []
for i in range(len(sup)):
    if sup[i]:
        used.append(X_Train.columns[i])
X_New = X_Train[used]
Test_New = X_Test[used]
clf = DecisionTreeClassifier()
clf.fit(X_New, Y_Train)
predict = clf.predict(Test_New)
results = pd.DataFrame({"PassengerId": X_Test['PassengerId'].astype('int'), "Survived": pd.Series(predict)})
results.to_csv('results.csv', index=False)
