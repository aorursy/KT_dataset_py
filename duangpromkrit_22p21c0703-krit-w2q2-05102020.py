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
#Import data, Drop unnecessary colums and Drop missing data
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train = df_train.drop(['PassengerId','Name', 'Cabin', 'Ticket'], axis=1)
df_train= df_train.dropna()
df_train.head()
#Categorical data -> numeric data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_train.Sex = le.fit_transform(df_train.Sex)
df_train.Embarked = le.fit_transform(df_train.Embarked)
df_train.head()
train = df_train.copy()
X = train.drop(['Survived'], axis=1).values
y = train['Survived'].values
train.head()
#split data -> train 80%,test 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 
#Decision Tree model
from sklearn.tree import DecisionTreeClassifier 
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz
export_graphviz(decision_tree, out_file='decision_tree.dot',  
                filled=True, rounded=True,
                special_characters=True)
!dot -Tpng decision_tree.dot -o decision_tree.png -Gdpi=200
from IPython.display import Image
Image(filename = 'decision_tree.png')
from sklearn.naive_bayes import GaussianNB #Naive Bayes model
from sklearn.neural_network import MLPClassifier #Neural network model

naive = GaussianNB()
naive.fit(X_train, y_train)

neural = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
neural.fit(X_train,y_train)
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

def evaluate_model(model, X_train, y_train, X_test, y_test):
    result = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    f1.append(f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return result, y_pred
kf= KFold(n_splits=5)
DTC= DecisionTreeClassifier()
NB= GaussianNB()
NN= MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
Model = [DTC,NB,NN]

for ML in Model:
    f1= []
    count =0
    print('Model : {}'.format(type(ML).__name__))
    for train_index, test_index in kf.split(X):
        count +=1
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
        print('fold : {0:0.0f}'.format(count))
        evaluate_model(ML, X_train, y_train, X_test, y_test)
    print('Average f1-score: {}'.format(np.mean(f1)))    
