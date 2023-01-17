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
import matplotlib.pyplot as plt;
import seaborn as sns
%matplotlib inline
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.info()
train_data.head()

test_data.info()
test_data.head()
#Convert Categorical data to numeric data
from sklearn import preprocessing
labelEnc = preprocessing.LabelEncoder()

train_data = train_data.drop(columns=['Name','Ticket', 'PassengerId', 'Cabin','Embarked'],axis=1)
train_data = train_data.dropna()

test_data = test_data.drop(columns=['Name','Ticket', 'PassengerId', 'Cabin','Embarked'],axis=1)
test_data = test_data.dropna()
train_data.head()
train_data.Sex = train_data.Sex.map({'male':0, 'female':1})
#train_data.Embarked = train_data.Embarked.map({'S':0, 'C':1, 'Q':2})

train_data.head()
test_data.Sex = test_data.Sex.map({'male':0, 'female':1})
#test_data.Embarked = test_data.Embarked.map({'S':0, 'C':1 ,'Q':2})

test_data.head()
train_data.isnull().sum()
train_data.Age.fillna(train_data.Age.mean(), inplace=True)
train_data.isnull().values.any()
test_data.isnull().sum()
y = train_data.Survived.copy()
X = train_data.drop(['Survived'], axis=1)
X.isnull().values.any()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=7) 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
decision_tree_Y_pred = decision_tree.predict(X_test)
decision_tree_accuracy = decision_tree.score(X_train, y_train)
print("Decision Tree Accuracy:" ,decision_tree_accuracy)
from sklearn.naive_bayes import GaussianNB #Naive Bayes model


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
guassian_Y_pred = gaussian.predict(X_test)
gaussian_accuracy = gaussian.score(X_train, y_train)
print("Gaussian Accuracy:" ,gaussian_accuracy)
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
from sklearn.metrics import precision_score, recall_score
from sklearn.neural_network import MLPClassifier


MLP_Model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
result = MLP_Model.fit(X_train,y_train)
y_pred = MLP_Model.predict(X_test)
print("Neural Network Accuracy:",metrics.accuracy_score(y_test, y_pred))

def evaluation(y_true, y_pred, pos_label=1):
    l = len(y_pred)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range (l):
        if y_pred[i] == pos_label: #tp,fp
            if y_pred[i] == y_true[i]:
                tp += 1
            else:
                fp +=1
        else:
            if y_pred[i] == y_true[i]:
                tn += 1
            else:
                fn +=1       
    p = tp / (tp+fp)
    r = tp/(tp+fn)
    f1 = 2*p*r/(p+r)
    return {"Precision": p, "Recall": r, "F1": f1}


def Evaluate_Model(model, X_train, Y_train, X_test, Y_test):
    result = model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    print(evaluation(Y_test, y_pred,1))
    print(evaluation(Y_test, y_pred,0))
    
from sklearn.model_selection import KFold

y = train_data.Survived.copy()
X = train_data.drop(['Survived'], axis=1)

X = np.array(X)
y = np.array(y)


k_fold = KFold(n_splits=5)
count = 1

for train_index, test_index in k_fold.split(X):
    
    print("====================")
    print("Fold NO => ", count )
    print("====================")
    
    count += 1
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    Y_train = Y_train.reshape(len(Y_train),)
    
    decision_tree = DecisionTreeClassifier()
    gaussian = GaussianNB()
    MLP_Model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    
    
    print("Decision Tree: ")
    Evaluate_Model(decision_tree, X_train, Y_train, X_test, Y_test)
    
    print("\nNaive Bays")
    Evaluate_Model(gaussian , X_train, Y_train, X_test, Y_test)
    
    
    print("\nNeural Network")
    Evaluate_Model(MLP_Model, X_train, Y_train, X_test, Y_test)
    
    
    print("\n")