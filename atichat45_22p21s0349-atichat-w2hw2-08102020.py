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

import numpy as np

import math
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.metrics import precision_score, recall_score
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
features = ['Pclass', 'Sex', 'SibSp', 'Age', 'Fare','Survived']

f = pd.get_dummies(train_data[features])

print(f)
data = f[['Pclass', 'SibSp', 'Fare', 'Sex_female', 'Sex_male','Survived']]

data.head(10)
def evaluation(y_true, y_pred, pos_label=1):

    l = len(y_pred)

    true_positive = 0

    false_positive = 0

    true_negative = 0

    false_negative = 0

    for i in range (l):

        if y_pred[i] == pos_label: #tp,fp

            if y_pred[i] == y_true[i]:

                true_positive += 1

            else:

                false_positive +=1

        else:

            if y_pred[i] == y_true[i]:

                true_negative += 1

            else:

                false_negative +=1       

    p = true_positive / (true_positive+false_positive)

    r = true_positive/(true_positive+false_negative)

    f1 = 2*p*r/(p+r)

    return {"precision": p, "recall": r, "f1": f1}





X = data[['Pclass', 'SibSp', 'Fare', 'Sex_female', 'Sex_male']]

Y = data[['Survived']]

X = np.array(X)

Y = np.array(Y)

print(X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 

y_test = np.array(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier
Dec_model = DecisionTreeClassifier()

result = Dec_model.fit(X_train,y_train)

predictions = Dec_model.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(evaluation(y_test, predictions,1))

print(evaluation(y_test, predictions,0))
NaBa_model = GaussianNB()

result = NaBa_model.fit(X_train,y_train)

predictions = NaBa_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(evaluation(y_test, predictions,1))

print(evaluation(y_test, predictions,0))
MLP_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

result = MLP_model.fit(X_train,y_train)

predictions = MLP_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(evaluation(y_test, predictions,1))

print(evaluation(y_test, predictions,0))
from sklearn.model_selection import KFold
def evaluate_model(model, X_train, Y_train, X_test, Y_test):

    result = model.fit(X_train,Y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

    print(evaluation(Y_test, y_pred,1))

    print(evaluation(Y_test, y_pred,0))
kf = KFold(n_splits=10)

count = 1

for train_index, test_index in kf.split(X):

    print("Fold =", count )

    count += 1

    X_train, X_test = X[train_index], X[test_index]

    Y_train, Y_test = Y[train_index], Y[test_index]

    Y_train = Y_train.reshape(len(Y_train),)



    Dec_model = DecisionTreeClassifier()

    NaBa_model = GaussianNB()

    MLP_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)



    print("Decision Tree: ")

    evaluate_model(Dec_model, X_train, Y_train, X_test, Y_test)

    print("Naive Bays")

    evaluate_model(NaBa_model, X_train, Y_train, X_test, Y_test)

    print("Neuron Network")

    evaluate_model(MLP_model, X_train, Y_train, X_test, Y_test)