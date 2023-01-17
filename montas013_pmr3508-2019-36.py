#imports



######################################################################################################



import numpy as np 

import pandas as pd 

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns



######################################################################################################



import sklearn

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import preprocessing 

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB



######################################################################################################



plt.rc("font", size=14)

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)



######################################################################################################

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
addtrain = '/kaggle/input/adult-pmr3508/train_data.csv' #train data address

addtest = '/kaggle/input/adult-pmr3508/test_data.csv' #test data address

addcatcol = ['workclass','education','marital.status','occupation','relationship','race','sex','native.country'] #columns with categorical values
#opening the train database

train_adult = pd.read_csv(addtrain,na_values='?')

clean_train_adult = train_adult.dropna()

train_adult = clean_train_adult

#opening the test adult database

test_adult = pd.read_csv(addtest,na_values='?')

clean_test_adult = test_adult.dropna()

test_adult = clean_test_adult
train_adult.head()
train_adult.income.value_counts()

sns.countplot(x = 'income',data = train_adult, palette = 'hls')

plt.title('Income Distribution')

plt.ylabel('Frequency')

plt.show()
train_adult.age.hist()

plt.title('Age Distribution')

plt.xlabel('Age')

plt.ylabel('Frequency')
%matplotlib inline

table = pd.crosstab(train_adult.occupation,train_adult.income)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Income Distribution over Occupation')

plt.xlabel('Occupation')

plt.ylabel('Frequency')
%matplotlib inline

pd.crosstab(train_adult.sex,train_adult.income).plot(kind='bar')

plt.title('Income Distribution over Sex')

plt.xlabel('Sex')

plt.ylabel('Frequency')
table = pd.crosstab(train_adult.relationship,train_adult.income)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Income Distribution over Relationship Status')

plt.xlabel('Relationship Status')

plt.ylabel('Frequency')
table=pd.crosstab(train_adult.education,train_adult.income)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Income Distribution over Level of Education')

plt.xlabel('Level of Education')

plt.ylabel('Frequency')
#encoding categorical data

train_adult = train_adult.apply(preprocessing.LabelEncoder().fit_transform)

test_adult = test_adult.apply(preprocessing.LabelEncoder().fit_transform)

#training and testing variables (these are the same for all classifiers)

Xtrain = train_adult.iloc[:,0:-1]

Ytrain = train_adult.income

Xtest = test_adult.iloc[:,0:]

#splitting the training set to measure model performance:

XStrain, XStest, YStrain, YStest = train_test_split(Xtrain, Ytrain, test_size=0.10, random_state=0)
#scaling data

scaler = StandardScaler()

scaler.fit(XStrain)



XStrain = scaler.transform(XStrain)

XStest = scaler.transform(XStest)
#training the parameter using the training data

logRegression = LogisticRegression()

logRegression = logRegression.fit(XStrain,YStrain)
#predictions

LRPredict = logRegression.predict(XStest)
#evaluating model

LRAccuracy = logRegression.score(XStest, YStest)

print('Accuracy Score: {0}% '.format(100*round(LRAccuracy,4)))
#classification report

LRClassification_Report = classification_report(YStest, LRPredict)

print("Classification Report")

print(LRClassification_Report)
#confusion matrix

LRConfusion_Matrix = confusion_matrix(YStest, LRPredict)

plt.figure(figsize=(5,5))

sns.heatmap(LRConfusion_Matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}%'.format(100*round(LRAccuracy,4))

plt.title(all_sample_title, size = 15);
#training the parameter using training data

DecTree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

DeCTree = DecTree.fit(XStrain,YStrain)
#prediction

DTPred = DeCTree.predict(XStest)
#evaluating model

DTAccuracy = metrics.accuracy_score(YStest,DTPred)

print('Accuracy Score: {0}% '.format(100*round(DTAccuracy,3)))
#classification report

DTClassification_Report = classification_report(YStest, LRPredict)

print("Classification Report")

print(DTClassification_Report)
#confusion matrix

DTConfusion_Matrix = confusion_matrix(YStest, DTPred)

plt.figure(figsize=(5,5))

sns.heatmap(LRConfusion_Matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}%'.format(100*round(DTAccuracy,3))

plt.title(all_sample_title, size = 15);
#training the parameter using training data

NB = GaussianNB()

NB = NB.fit(XStrain,YStrain)
#predictions

NBPred = NB.predict(XStest)
#evaluating model

NBAccuracy = metrics.accuracy_score(YStest,NBPred)

print('Accuracy Score: {0}% '.format(100*round(NBAccuracy,3)))
#classification report

NBClassification_Report = classification_report(YStest, NBPred)

print("Classification Report")

print(NBClassification_Report)
#confusion matrix

NBConfusion_Matrix = confusion_matrix(YStest, NBPred)

plt.figure(figsize=(5,5))

sns.heatmap(NBConfusion_Matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}%'.format(100*round(NBAccuracy,3))

plt.title(all_sample_title, size = 15);