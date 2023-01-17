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
set_dummy = [pd.get_dummies(df[e],drop_first=True) for e in ['Sex','Embarked'] ]
Ignore_feature = ['Name','Ticket','Cabin','Sex','Embarked']
X = pd.concat([df.drop(['Survived']+Ignore_feature,axis=1)]+set_dummy , axis=1)
y = df['Survived']
Drop_ind = []
c= 0
for i in range(len(X)):
    if np.isnan(X['Age'][i]):
        Drop_ind += [i]
X = X.drop(Drop_ind)
y = y.drop(Drop_ind)
        
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

c = 1
F1_avg = 0.0
print('5 Fold with Decision Tree')
for train_index, test_index in kf.split(X):
    X_tr,y_tr,X_ts,y_ts = X.loc[train_index,:],y.loc[train_index],X.loc[test_index,:],y.loc[test_index]
    print('Decision Tree Fold',c);c+=1
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_tr, y_tr)
    Predicts = tree.predict(X_ts)
    print(classification_report(y_ts,Predicts))
    Reports = classification_report(y_ts,Predicts,output_dict=True)
    F1_avg += Reports['macro avg']['f1-score']
F1_avg = F1_avg/5
print('F1-score average of 5 fold(Macro F1) =',F1_avg)
from sklearn.naive_bayes import GaussianNB

c = 1
F1_avg = 0.0
print('5 Fold with Gaussian Naive Bayes')
for train_index, test_index in kf.split(X):
    X_tr,y_tr,X_ts,y_ts = X.loc[train_index,:],y.loc[train_index],X.loc[test_index,:],y.loc[test_index]
    print('Naive Bayes Fold',c);c+=1
    clf = GaussianNB()
    clf.fit(X_tr, y_tr)
    Predicts = clf.predict(X_ts)
    print(classification_report(y_ts,Predicts))
    Reports = classification_report(y_ts,Predicts,output_dict=True)
    F1_avg += Reports['macro avg']['f1-score']
F1_avg = F1_avg/5
print('F1-score average of 5 fold(Macro F1) =',F1_avg)
from sklearn.neural_network import MLPClassifier
print('5 Fold with Multi-layer Perceptron (Neural network models )')
c=1
F1_avg = 0.0
for train_index, test_index in kf.split(X):
    X_tr,y_tr,X_ts,y_ts = X.loc[train_index,:],y.loc[train_index],X.loc[test_index,:],y.loc[test_index]
    print('Neural network Fold',c);c+=1
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_tr, y_tr)
    Predicts = clf.predict(X_ts)
    print(classification_report(y_ts,Predicts))
    Reports = classification_report(y_ts,Predicts,output_dict=True)
    F1_avg += Reports['macro avg']['f1-score']
F1_avg = F1_avg/5
print('F1-score average of 5 fold(Macro F1) =',F1_avg)
