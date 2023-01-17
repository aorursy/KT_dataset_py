import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
#dataset read operation

#read_csv function is required to read the data.

file=pd.read_csv('../input/winequality-red.csv')

file.head()
#count of columns

len(file.columns)
# columns names

file.columns
#count value of gender column

file['quality'].value_counts()
#to know the na values in each column

file.isna().sum() #no na values
#information about file

file.info()
#median

file.median()
file.var()
file.std()
file.skew()
file.kurtosis()
file.hist()
file.boxplot()
#scatter plot

import seaborn as sns

sns.pairplot(file)
# Correlation matrix 

file.corr()
bins = (2,6.5,8)

group_names=['bad','good']

file['quality']=pd.cut(file['quality'],bins=bins,labels=group_names)
file.head()
#Correlation matrix 

file.corr()
# Consider the inpur varabile as X and Output variable as Y 

X = file[['fixed acidity','citric acid','residual sugar','sulphates','alcohol']]

Y = file[['quality']]

print(X.shape)

print(Y.shape)

print(X)

print(Y)
norm=(X-X.min())/(X.max()-X.min())

norm
#KNN

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)  

classifier.fit(norm,Y)
#prediction

knnpred= classifier.predict(norm)

knnpred
#confusion matrix

from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(Y, knnpred))
#accuracy

from sklearn.metrics import accuracy_score 

Accuracy_Score = accuracy_score(Y, knnpred)

Accuracy_Score
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

model.fit(norm,Y)
# Predicting the Model

nbpred = model.predict(norm)

nbpred
##Evaluating the Algorithm (confusion matrix)

from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(Y, nbpred))
#accuracy

from sklearn.metrics import accuracy_score 

Accuracy_Score = accuracy_score(Y, nbpred)

Accuracy_Score
#decision tree

# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(norm, Y)
dtpred=classifier.predict(norm)

dtpred
#confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y,dtpred)

cm
#accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y,dtpred)

accuracy
#Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(norm,Y)
#prediction

rfpred=classifier.predict(norm)

rfpred
#confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y,rfpred)

cm
#accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y,rfpred)

accuracy
#SVM model

from sklearn.svm import SVC

cls=SVC(kernel='linear',random_state=0)

cls.fit(norm,Y)
#prediction

svmpred=cls.predict(norm)

svmpred
#confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y,svmpred)

cm
#accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y,svmpred)

accuracy
#SVM non-linear kernel method

from sklearn.svm import SVC

cls=SVC(kernel='sigmoid',random_state=0)

cls.fit(norm,Y)
#predict

svmpred1=cls.predict(norm)

svmpred1
#confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y,svmpred1)

cm
#accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y,svmpred1)

accuracy
#to build the logistic regression model on train dataset

from sklearn.linear_model import LogisticRegression  

regressor = LogisticRegression()  

logit=regressor.fit(norm, Y)
print(logit.intercept_,logit.coef_) 
np.exp(logit.intercept_,logit.coef_)
#prediction

lnpred = logit.predict(norm)

lnpred 
#package to build confusion matrix

from sklearn import metrics

cm = metrics.confusion_matrix(Y , lnpred)

print(cm)
#accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y,lnpred)

accuracy