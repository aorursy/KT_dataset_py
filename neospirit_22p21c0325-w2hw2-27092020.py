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
import pandas as pd
train = pd.read_csv("../input/titanic/train.csv")

survived_test = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
test = pd.merge(test, survived_test, on='PassengerId')
train
test
print(train.isna().sum())
train = train.copy().dropna(subset=['Age', 'Embarked'])
train.drop(['Cabin','Ticket'], axis=1, inplace=True)
train = train.reset_index()

test.drop(['Cabin','Ticket'], axis=1, inplace=True)
test = test.reset_index()
train
test
# Get only surname for finding isFamily.
train['Surname'] = train.Name.copy()
for index,name in enumerate(train.Surname):
    surname = name.split(",")[0]
    train.Surname[index] = surname
    
# Encode Sex as number (female=0, male=1)    
train["Sex"] = train["Sex"].apply(lambda row: int(row == "male"))

# Set new index as PassengerID
train.index = train.PassengerId	
train.drop(['index','PassengerId','Name'],axis=1, inplace=True)
# Get only surname for finding isFamily.
test['Surname'] = test.Name.copy()
for index,name in enumerate(test.Surname):
    surname = name.split(",")[0]
    test.Surname[index] = surname
    
# Encode Sex as number (female=0, male=1)    
test["Sex"] = test["Sex"].apply(lambda row: int(row == "male"))

# Set new index as PassengerID
test.index = test.PassengerId	
test.drop(['index','PassengerId','Name'],axis=1, inplace=True)
train
test
import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split,KFold
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.metrics import classification_report
col = train.columns
X = train[col[1:6]]
Y = train[col[0]]
X
Y
X = np.array(X)
Y = np.array(Y)

kfold = KFold(n_splits=5, random_state=42, shuffle=False)
decisionTree = DecisionTreeClassifier(criterion="entropy", max_depth=None)

for count, (train_index, valid_index) in enumerate(kfold.split(X)):
    x_train = X[train_index]
    y_train = Y[train_index]
    x_test = X[valid_index]
    y_test = Y[valid_index]

    decisionTree = decisionTree.fit(x_train,y_train)
    
    y_pred = decisionTree.predict(x_test)
    y_score = decisionTree.score(x_test,y_test)
    
    print(f'       Decision Tree Model: Fold {count+1}\n')
    print(f'              Train Fold {count} score :', y_score)
    print(classification_report(y_test,y_pred)+"\n")

y_pred = decisionTree.predict(X)
y_score = decisionTree.score(X,Y)
print('       Decision Tree Model: Final\n')
print('              Final evaluation score :'+str(y_score)+'\n')
print(classification_report(Y,y_pred))
graph = Source(export_graphviz(decisionTree, out_file=None,  
                filled=True, rounded=True,
                special_characters=True,feature_names = col[1:6],class_names=['0','1','2']))

display(SVG(graph.pipe(format='svg')))
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()

for count, (train_index, valid_index) in enumerate(kfold.split(X)):
    x_train = X[train_index]
    y_train = Y[train_index]
    x_test = X[valid_index]
    y_test = Y[valid_index]

    model_naive = naive.fit(x_train,y_train)
    
    y_pred = model_naive.predict(x_test)
    y_score = model_naive.score(x_test,y_test)
    
    print(f'       Naive Bayes Model: Fold {count+1}\n')
    print(f'              Train Fold {count+1} score :', y_score)
    print(classification_report(y_test,y_pred)+"\n")

y_pred = model_naive.predict(X)
y_score = model_naive.score(X,Y)
print('       Decision Tree Model: Final\n')
print('              Final evaluation score :'+str(y_score)+'\n')
print(classification_report(Y,y_pred))
from sklearn.neural_network import MLPClassifier
neural = MLPClassifier(hidden_layer_sizes=(216,), random_state=42, max_iter=1, warm_start=True)

for count, (train_index, valid_index) in enumerate(kfold.split(X)):
    x_train = X[train_index]
    y_train = Y[train_index]
    x_test = X[valid_index]
    y_test = Y[valid_index]

    for i in range(2000):
        neural.fit(x_train, y_train)
    
    y_pred = neural.predict(x_test)
    y_score = neural.score(x_test,y_test)
    
    print(f'       Neural Network Model(MLP): Fold {count+1}\n')
    print(f'              Train Fold {count+1} score :', y_score)
    print(classification_report(y_test,y_pred)+"\n")

y_pred = neural.predict(X)
y_score = neural.score(X,Y)
print('       Decision Tree Model: Final\n')
print('              Final evaluation score :'+str(y_score)+'\n')
print(classification_report(Y,y_pred))
test
test = test.fillna(0)

test_col = test.columns
test_data = test[col[1:6]]
test_result = test.Survived
test_pred = neural.predict(test_data)

print(classification_report(test_result,test_pred))
test_pred
survived_test.Survived = test_pred
survived_test
survived_test.to_csv('submission.csv',index=False)