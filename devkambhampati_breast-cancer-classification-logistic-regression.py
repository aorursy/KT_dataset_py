import numpy as np

import pandas as pd

from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#Importing BREAST CANCER DATA FROM SKLEARN DATASETS

data = load_breast_cancer() 

data.keys()  
data.feature_names  #column names

data.feature_names
data.data[0:2] #actual data- numpy array
data.target[0:20]  #target array
data.DESCR  #description of dataset
data.target_names
BCdf=pd.DataFrame(data.data,columns=data.feature_names)

BCdf.size
BCdf.shape   #569 rows, 30 columns
BCdf.head(3)

BCdf['Target']=data.target
BCdf.head()
BCdf.describe()
BCdf.info
BCdf.dtypes
#IMPORTING LIBRARIES FROM SKLEARN

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
X=data.data

y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y)
#Create an instance of the model

LR=LogisticRegression()
LR
#Fit the Model to the Data



LR.fit(X_train,y_train)
#MODEL ACCURACY

LR.score(X_test,y_test)  # Generates 93% accuracy using LOGISTIC REGRESSION MODEL

#Generated accuracy of 93%
yhat=LR.predict(X_test)   #Predicts labels for test data

yhat
yhat_prob=LR.predict_proba(X_test)   #Predicts probability,useful also in log loss calculations

yhat_prob


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test,yhat)  #93% accuracy
from sklearn.metrics import classification_report, confusion_matrix

CM=metrics.confusion_matrix(y_test,yhat,labels=[0,1])
print(CM)
plt.figure(figsize=(6,6))

sns.heatmap(CM,annot=True)
classification_report(y_test,yhat)

#Shows PRECISION, RECALL, F1 SCORE
from sklearn.metrics import log_loss
log_loss(y_test,yhat_prob)  #Log Loss for Logistic Regression Classifier is 0.1888