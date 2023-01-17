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
! pip install dexplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot
df = pd.read_csv('/kaggle/input/lung-cancer-dataset/lung_cancer_examples.csv')
df.head()
# Trying to get the info from the dataset
df.info()
# # Trying to see whether there is any null value or bot in the dataset
df.isnull().sum()
# Lets describe the dataset

df.describe()
# the number of unique values in Smokes columns 
print(df['Smokes'].nunique())
df['Smokes'].unique()
# the number of unique values in AreaQ columns
print(df['AreaQ'].nunique())
df['AreaQ'].unique()
# the number of unique values in Alkhol columns
print(df['Alkhol'].nunique())
df['Alkhol'].unique()
# the number of unique values in Age columns
print(df['Age'].nunique())
df['Age'].unique()
# Since Name and Surname are in no way impacting the accuracy of the model so Dropping them

df.drop(['Name', 'Surname'], inplace = True, axis = 'columns')
df.head()
# Lets see the distribution of Data about the persons who are suffering from 
# Lung Cancer or Not

labels = df['Result'].value_counts()[:].index
values = df['Result'].value_counts()[:].values

colors=['#2678bf', '#98adbf']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()

# Lets visualize the distribution of the person who are smokes in 
# terms of the number cigrates smokes

# I will plot here the bar graph for the top 10 smokers

labels = df['Smokes'].value_counts()[:10].index
values = df['Smokes'].value_counts()[:10].values

colors=df['Smokes']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
# Lets visualize the AreaQ 
labels = df['AreaQ'].value_counts().index
values = df['AreaQ'].value_counts().values

colors=df['AreaQ']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
# Lets visualize the Alkhol

labels = df['Alkhol'].value_counts().index
values = df['Alkhol'].value_counts().values

colors=df['Alkhol']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
# Lets visualize the number of cigarattes one Smoke and whether he suffers from 
# Lung Cancer or not

import dexplot as dxp

dxp.count(val='Smokes', data=df, figsize=(4,3), split = 'Result', normalize=True)
# Splitting the data

from sklearn.model_selection import train_test_split

X = df.drop(['Result'], axis = 'columns')
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
# Printing the shape of the splitted data

print('The shape of X_train is {}'.format(X_train.shape))
print('The shape of X_test is {}'.format(X_test.shape))
print('The shape of y_train is {}'.format(y_train.shape))
print('The shape of y_test is {}'.format(y_test.shape))
def true_positive(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: number of true positives
    """
    
    # initialize
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: number of true negatives
    """
    
    # initialize
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: number of false positives
    """
    
    # initialize
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: number of true positives
    """
    
    # initialize
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()
model_log.fit(X_train, y_train)
predict1 = model_log.predict(X_test)
# Defining the model

def accuracy_score(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: accuracy score
    """
    
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    
    accuracy_score = (tp+tn)/(tp+tn+fp+fn)
    return accuracy_score
# Calculating the accuracy score of the above model

accuracy_score(y_test, predict1)
# Definig the model

def precision(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: precision score
    """
    
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision = tp/(tp+fp)
    return precision
# Calculating the precision score of the above model


precision(y_test, predict1)
# Defining the model

def recall(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: recall score
    """
    
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp/(tp+fn)
    return recall
# Calculating the recall score of the above model


recall(y_test, predict1)
# Defining the model

def f1(y_true, y_pred):
    """
    Function to calculate the True Positive
    : param y_true: list of true values
    : param y_pred: list of predicted values
    : return: f1 score
    """
    
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    score = 2*p*r/(p+r)
    
    return score
# Calculating the f1 score 

f1(y_test, predict1)
# Calculating the roc_auc_score

from sklearn import metrics
metrics.roc_auc_score(y_test, predict1)
# Plotting the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict1)
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot = True, ax = ax)
plt.title('Confusion Matrix for the Logistic Regression')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()
from sklearn.tree import DecisionTreeClassifier

model_dtc = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                   max_features=None, random_state=None, max_leaf_nodes=None, 
                                   min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, 
                                   presort='deprecated', ccp_alpha=0.0)
model_dtc.fit(X_train, y_train)
predict2 = model_dtc.predict(X_test)
# Calculating the accuracy score of the above model

accuracy_score(y_test, predict2)
# Calculating the precision score of the above model


precision(y_test, predict2)
# Calculating the2recall score of the above model


recall(y_test, predict2)
# Calculating the f1 score 

f1(y_test, predict2)
# Calculating the roc_auc_score

from sklearn import metrics
metrics.roc_auc_score(y_test, predict2)
# Plotting the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict2)
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot = True, ax = ax)
plt.title('Confusion Matrix for the Decision Tree Classification')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()
from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
model_rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                   bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 
                                   verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
model_rfc.fit(X_train, y_train)
predict3 = model_rfc.predict(X_test)
# Calculating the accuracy score of the above model

accuracy_score(y_test, predict3)
# Calculating the precision score of the above model


precision(y_test, predict3)
# Calculating the recall score of the above model


recall(y_test, predict3)
# Calculating the f1 score 

f1(y_test, predict3)
# Calculating the roc_auc_score

from sklearn import metrics
metrics.roc_auc_score(y_test, predict3)
# Plotting the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict3)
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot = True, ax = ax)
plt.title('Confusion Matrix for the Decision Tree Classification')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()
#Import svm model
from sklearn import svm

#Create a svm Classifier
model_svm = svm.SVC(kernel='linear') # Linear Kernel

model_svm.fit(X_train, y_train)
predict4 = model_svm.predict(X_test)
# Calculating the accuracy score of the above model

accuracy_score(y_test, predict4)
# Calculating the precision score of the above model


precision(y_test, predict4)
# Calculating the recall score of the above model


recall(y_test, predict4)
# Calculating the f1 score 

f1(y_test, predict4)
# Calculating the roc_auc_score

from sklearn import metrics
metrics.roc_auc_score(y_test, predict4)
# Plotting the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict4)
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot = True, ax = ax)
plt.title('Confusion Matrix for the Decision Tree Classification')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model_NB = GaussianNB()

# Train the model using the training sets
model_NB.fit(X_train, y_train)
predict5 = model_NB.predict(X_test)
# Calculating the accuracy score of the above model

accuracy_score(y_test, predict5)
# Calculating the prec5sion score of the above model


precision(y_test, predict5)
# Calcul5ting the recall score of the above model


recall(y_test, predict5)
# Calculating the f1 score 

f1(y_test, predict5)
# Calculating the roc_auc_score

from sklearn import metrics
metrics.roc_auc_score(y_test, predict5)
# Plotting the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict5)
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot = True, ax = ax)
plt.title('Confusion Matrix for the Decision Tree Classification')
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
plt.show()
