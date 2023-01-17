# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/titanic/train.csv")

df
X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']].fillna(0)

y = df[['Survived']].fillna(0)



feature_names = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']



X.loc[X['Sex'] == 'male', 'Sex'] = 0  

X.loc[X['Sex'] == 'female', 'Sex'] = 1 

X['Sex'] = X['Sex'].astype(str).astype(int)



X.loc[X['Cabin'] != 0, 'Cabin'] = 1 

X['Cabin'] = X['Cabin'].astype(str).astype(int)



X.loc[X['Embarked'] == 'S', 'Embarked'] = 1 

X.loc[X['Embarked'] == 'C', 'Embarked'] = 2

X.loc[X['Embarked'] == 'Q', 'Embarked'] = 3 

X['Embarked'] = X['Embarked'].astype(str).astype(int)



y = y['Survived']
df_test = pd.read_csv("../input/titanic/test.csv")

df_test
X_val = df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']].fillna(0)



X_val.loc[X_val['Sex'] == 'male', 'Sex'] = 0  

X_val.loc[X_val['Sex'] == 'female', 'Sex'] = 1 

X_val['Sex'] = X_val['Sex'].astype(str).astype(int)



X_val.loc[X_val['Cabin'] != 0, 'Cabin'] = 1 

X_val['Cabin'] = X_val['Cabin'].astype(str).astype(int)



X_val.loc[X_val['Embarked'] == 'S', 'Embarked'] = 1 

X_val.loc[X_val['Embarked'] == 'C', 'Embarked'] = 2

X_val.loc[X_val['Embarked'] == 'Q', 'Embarked'] = 3 

X_val['Embarked'] = X_val['Embarked'].astype(str).astype(int)



#X_val = X_val.values.tolist()

len(X_val)
df_val = pd.read_csv("../input/titanic/gender_submission.csv")

df_val
y_val = df_val[['Survived']].fillna(0)

y_val = y_val['Survived']

len(y_val)
# Decision Tree Classifier



from sklearn import tree

from sklearn.tree import export_text



clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, y)

tree.plot_tree(clf) 



#r = tree.export_text(clf, feature_names=feature_names)

#print(r)



print(f'predict:{clf.predict(X_val)}')

print(f'score:{clf.score(X_val, y_val)}')
# Naïve Bayes Classifier



from sklearn.naive_bayes import GaussianNB



clf = GaussianNB()

clf.fit(X, y)



print(f'predict:{clf.predict(X_val)}')

print(f'score:{clf.score(X_val, y_val)}')
# Neural Network Classifier (Multi-layer Perceptron classifier)



from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split



clf = MLPClassifier(random_state=1, max_iter=300).fit(X, y)



print(f'predict:{clf.predict(X_val)}')



print(f'score:{clf.score(X_val, y_val)}')
# 5-fold cross validation



from sklearn.model_selection import cross_validate

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score



clf = GaussianNB()



scores = cross_validate(clf, X, y, cv=5, scoring=('recall', 'precision', 'f1'),return_estimator=True)



print(f"     class:       1          2          3          4          5")

print(f"    recall: {scores['test_recall']}")

print(f" precision: {scores['test_precision']}")

print(f"  f1_score: {scores['test_f1']}")

print()



f1s = []

for est in scores['estimator']:

    p = est.predict(X)

    f1s.append(f1_score(y, p, average='macro'))

    

print(f"    all f1: {f1s}")

print(f"average f1: {sum(f1s) / len(f1s)}")
