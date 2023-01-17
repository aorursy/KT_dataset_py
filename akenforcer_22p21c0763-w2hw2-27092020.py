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
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df
[ (col, df.isna()[col].sum()) for col in df.columns ]

ignoreFeatures = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
df = df.drop(ignoreFeatures, axis = 1)
df
df["Sex"] = df["Sex"].apply(lambda x:{"male":0, "female":1}[x])
df
df = df.fillna(df.mean())
df
x_train = df.drop(["Survived"], axis = 1)
x_train
y_train = df["Survived"]
y_train
k_folds = 5
from sklearn.model_selection import KFold
kf = KFold(n_splits=k_folds)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

c = 1
F1_avg = 0.0
print('5 Fold with Decision Tree')
for train_index, test_index in kf.split(x_train):
    xtr, ytr, xts, yts = x_train.loc[train_index,:], y_train.loc[train_index], x_train.loc[test_index,:], y_train.loc[test_index]
    print('Decision Tree Fold',c);c+=1
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(xtr, ytr)
    Predicts = tree.predict(xts)
    print(classification_report(yts,Predicts))
    Reports = classification_report(yts,Predicts,output_dict=True)
    F1_avg += Reports['macro avg']['f1-score']
F1_avg = F1_avg/5
print('F1-score average of 5 fold(Macro F1) =',F1_avg)
from sklearn.naive_bayes import GaussianNB

c = 1
F1_avg = 0.0
print('5 Fold with Gaussian Naive Bayes')
for train_index, test_index in kf.split(x_train):
    xtr, ytr, xts, yts = x_train.loc[train_index,:], y_train.loc[train_index], x_train.loc[test_index,:], y_train.loc[test_index]
    print('Naive Bayes Fold',c);c+=1
    clf = GaussianNB()
    clf.fit(xtr, ytr)
    Predicts = clf.predict(xts)
    print(classification_report(yts,Predicts))
    Reports = classification_report(yts,Predicts,output_dict=True)
    F1_avg += Reports['macro avg']['f1-score']
F1_avg = F1_avg/5
print('F1-score average of 5 fold(Macro F1) =',F1_avg)
from sklearn.neural_network import MLPClassifier
print('5 Fold with Multi-layer Perceptron (Neural network models )')
c=1
F1_avg = 0.0
for train_index, test_index in kf.split(x_train):
    X_tr,y_tr,X_ts,y_ts = x_train.loc[train_index,:], y_train.loc[train_index], x_train.loc[test_index,:], y_train.loc[test_index]
    print('Neural network Fold',c);c+=1
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6, 2), random_state=1)
    clf.fit(X_tr, y_tr)
    Predicts = clf.predict(X_ts)
    print(classification_report(y_ts,Predicts))
    Reports = classification_report(y_ts,Predicts,output_dict=True)
    F1_avg += Reports['macro avg']['f1-score']
F1_avg = F1_avg/5
print('F1-score average of 5 fold(Macro F1) =',F1_avg)