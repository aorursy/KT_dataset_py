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
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df1 = pd.read_csv('/kaggle/input/titanic/train.csv')
df1.head()

from pandas_profiling import ProfileReport
pf = ProfileReport(df1, title="Rep1")
pf.to_widgets()
dt = df1[['Pclass', 'Sex', 'SibSp', 'Age', 'Fare','Survived']]
dt.head(10)

def clean_dt(dt):
    dt['AgeRange'] = dt['Age'].apply(lambda x: int(x/20) if not math.isnan(x) else x)
    dt['Male'] = dt['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    dt['Female'] = dt['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    dt['Survived'] = dt['Survived'].apply(lambda x: 1 if x else 0)
    return dt
dt = clean_dt(dt)
dt = dt.replace(np.nan, 0)
dt.head(20)

profile2 = ProfileReport(dt, title="Rep 2")
dt = dt[['Pclass', 'SibSp', 'Fare', 'Male', 'Female', 'Survived']]
dt.head()
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
from sklearn.metrics import precision_score, recall_score

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
    return {"precision": p, "recall": r, "f1": f1}


X = dt[['Pclass', 'SibSp', 'Fare', 'Male', 'Female']]
Y = dt[['Survived']]
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 
y_test = np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# Train Decision Tree Classifer
result = model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(evaluation(y_test, y_pred,1))
print(evaluation(y_test, y_pred,0))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# Train Decision Tree Classifer
result = model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.naive_bayes import GaussianNB
bmodel = GaussianNB()
result = bmodel.fit(X_train,y_train)
y_pred = bmodel.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(evaluation(y_test, y_pred,1))
print(evaluation(y_test, y_pred,0))

from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
result = mlp_model.fit(X_train,y_train)
y_pred = mlp_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(evaluation(y_test, y_pred,1))
print(evaluation(y_test, y_pred,0))

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    result = model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    print(evaluation(Y_test, y_pred,1))
    print(evaluation(Y_test, y_pred,0))
    
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
count = 1
for train_index, test_index in kf.split(X):
    print("Fold: ", count )
    count += 1
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    Y_train = Y_train.reshape(len(Y_train),)
    dmodel = DecisionTreeClassifier()
    bmodel = GaussianNB()
    mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    print("Dec Tree: ")
    evaluate_model(dmodel, X_train, Y_train, X_test, Y_test)
    print("Na Bays")
    evaluate_model(bmodel, X_train, Y_train, X_test, Y_test)
    print("Neu Network")
    evaluate_model(mlp_model, X_train, Y_train, X_test, Y_test)
