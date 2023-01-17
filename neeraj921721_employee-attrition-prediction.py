import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/dataset-nj/train.csv')

test = pd.read_csv('/kaggle/input/dataset-nj/test.csv')

sample_output = pd.read_csv('/kaggle/input/dataset-nj/Sample_submission.csv')
train.head()
test.head()
sample_output.head()
train.describe()
test.describe()
sample_output.describe()
Y = train['Attrition']

X = train

X = X.drop(['Id','Attrition','EmployeeNumber'],axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
# Label Encoding for Labelled data

for col in X.columns:

    if(isinstance(train[col][0],str)):

        X[col] = LabelEncoder().fit_transform(X[col])

        

for col in X_train.columns:

    if(isinstance(train[col][0],str)):

        X_train[col] = LabelEncoder().fit_transform(X_train[col])

        

for col in X_test.columns:

    if(isinstance(train[col][0],str)):

        X_test[col] = LabelEncoder().fit_transform(X_test[col])
X_test_data = test

X_test_data = X_test_data.drop(['Id','EmployeeNumber'],axis=1

                )

# Label Encoding for Labelled data

for col in X_test_data.columns:

    if(isinstance(test[col][0],str)):

        X_test_data[col] = LabelEncoder().fit_transform(X_test_data[col])
# train the model



dtc = DecisionTreeClassifier(max_depth=24,random_state=1)

dtc.fit(X_train,y_train)
# auc-roc score



print('For Decision Tree Classifier : ')

train_score = roc_auc_score(y_train,dtc.predict(X_train))

print('Train auc-roc score : ',train_score)

test_score = roc_auc_score(y_test,dtc.predict(X_test))

print('Test auc-roc score : ',test_score)
# train the model



rfc = RandomForestClassifier(n_estimators=1000,max_features=24,random_state=1)

rfc.fit(X_train,y_train)
# auc-roc score



print('For Randon Forest Classifier : ')

train_score = roc_auc_score(y_train,rfc.predict(X_train))

print('Train auc-roc score : ',train_score)

test_score = roc_auc_score(y_test,rfc.predict(X_test))

print('Test auc-roc score : ',test_score)
# train the model



gbc = GradientBoostingClassifier(n_estimators=100,max_features=24,random_state=0,learning_rate=1)

gbc.fit(X_train,y_train)
# auc-roc score



print('For Gradient Boosting Classifier : ')

train_score = roc_auc_score(y_train,gbc.predict(X_train))

print('Train auc-roc score : ',train_score)

test_score = roc_auc_score(y_test,gbc.predict(X_test))

print('Test auc-roc score : ',test_score)
# train the model



svc = SVC()

svc.fit(X_train,y_train)
# auc-roc score



print('For Support Vector Classifier : ')

train_score = roc_auc_score(y_train,svc.predict(X_train))

print('Train auc-roc score : ',train_score)

test_score = roc_auc_score(y_test,svc.predict(X_test))

print('Test auc-roc score : ',test_score)
# train the model



lrc = LogisticRegression(C=1,max_iter=1000)

lrc.fit(X_train,y_train)
# auc-roc score



print('For Logistic Regression Classifier : ')

train_score = roc_auc_score(y_train,lrc.predict(X_train))

print('Train auc-roc score : ',train_score)

test_score = roc_auc_score(y_test,lrc.predict(X_test))

print('Test auc-roc score : ',test_score)
# train the model



mlpc = MLPClassifier(random_state=0,activation='logistic',max_iter=300,hidden_layer_sizes=(10000,))

mlpc.fit(X_train,y_train)
# auc-roc score



print('For MultiLayer Perceptron Classifier : ')

train_score = roc_auc_score(y_train,mlpc.predict(X_train))

print('Train auc-roc score : ',train_score)

test_score = roc_auc_score(y_test,mlpc.predict(X_test))

print('Test auc-roc score : ',test_score)
models = [dtc, rfc, gbc, lrc, mlpc]

modelName = ['DecisionTreeClassifier', 

             'RandomForestClassifier', 

             'GradientBoostingClassifier', 

             'LogisticRegressionClassifier',

             'MultiLayerPrecetronClassifier'

            ]

for model in models:

    model.fit(X,Y)
for model,name in zip(models,modelName):

    test_pred = model.predict_proba(X_test_data)[:,1]

    model_result = pd.DataFrame({'Id':list(test['Id']),'Attrition':list(test_pred)})

    model_result.to_csv("/kaggle/working/"+str(name)+".csv",index=False)