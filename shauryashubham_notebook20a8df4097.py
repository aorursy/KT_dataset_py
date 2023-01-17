import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt
data=pd.read_csv('../input/creditcard.csv')

data.shape
data.head()
data.info()
data.describe()
print("Null values in features of data:")

data.isnull().sum()
data_labels=data['Class']

data = data.drop('Class',axis=1)

data.shape
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(data)
pd.DataFrame(X).describe()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, data_labels, test_size=0.25, random_state=0)
X_train.shape
from sklearn import metrics
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print('SVM Accuracy: ', metrics.accuracy_score(y_test,y_pred))
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Log Regression Accuracy: ', metrics.accuracy_score(y_test,y_pred))
import tflearn
y_train.shape
input_layer=tflearn.input_data([None,30])

g = tflearn.fully_connected(input_layer, 15, activation='relu')

g= tflearn.dropout(g,0.8)

g = tflearn.fully_connected(g,7, activation='relu')

g= tflearn.dropout(g,0.8)

g = tflearn.fully_connected(g, 2, activation='sigmoid')

sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)

g = tflearn.regression(g, optimizer=sgd, learning_rate=0.02,loss='categorical_crossentropy')
y_train=pd.get_dummies(y_train)

y_train=y_train.values
y_train.shape
X_train.shape
# Training

model = tflearn.DNN(g, tensorboard_verbose=0)

model.fit(X_train, y_train, n_epoch=40, validation_set=0.20,show_metric=True,batch_size=150)