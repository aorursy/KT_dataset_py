import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from pandas import DataFrame
# Fix random seed for reproducibility

np.random.seed(10)

plt.rcParams["figure.figsize"]=(10,5)
train=pd.read_csv('../input/titanic/train.csv')
# TOP 4 ROWS OF TRAIN DATA

train.head(4)
test=pd.read_csv('../input/titanic/test.csv')
# TOP 4 ROWS OF TEST DATA

test.head(4)
# Information on the data

train.info()
test.info()
train.isnull().sum()
train["Cabin"].fillna(method="bfill",inplace=True)

train["Cabin"].fillna(method="ffill",inplace=True)
train["Age"].fillna(value=train.Age.mean(),inplace=True)
train["Embarked"].mode()
train["Embarked"].fillna("S",inplace=True)
test.isnull().sum()
test["Age"].fillna(value=test.Age.mean(),inplace=True)
test["Fare"].interpolate(method="polynomial",order=2,inplace=True)
test["Cabin"].fillna(method="bfill",inplace=True)

test["Cabin"].fillna(method="ffill",inplace=True)
train.isnull().sum()
test.isnull().sum()
# Plot number of counts per column

list1=['Pclass','Sex', 'Embarked','SibSp','Parch','Survived']

for column in list1:

    plt.title(column)

    sns.countplot(column,data=train)

    plt.show()
# Minimum and Maximum age

train.Age.max(),test.Age.max(),train.Age.min(),test.Age.min()
#Binning Age

bins=np.arange(0,90,10)

train["Bin_Age"],test["Bin_Age"]=pd.cut(train["Age"],bins=bins),pd.cut(test["Age"],bins=bins)

train["Bin_Age"],test["Bin_Age"]=train["Bin_Age"].astype("category"),test["Bin_Age"].astype("category")
# Change to data type to category

int_cols=['Pclass','Embarked','SibSp','Parch','Bin_Age']

for cols in int_cols:

    train[cols]=train[cols].astype("category")
# Plot bar graph against survive

for columns in int_cols:

    ax=sns.barplot(columns,train["Survived"],hue='Sex',data=train)

    ax.set_title(columns)

    plt.show()

    
from sklearn import preprocessing

encode=preprocessing.LabelEncoder()
cols=['Embarked','Sex','Bin_Age','Cabin']

for column in cols:

    train[column]=encode.fit_transform(train[column])

    test[column]=encode.fit_transform(test[column])
# Data Types

train.dtypes
train.head(4)
test.head(4)
# Summary Statistics

train.describe()
# Correlation between attributes

print(train.corr())

sns.heatmap(train.corr())

plt.show()
x_train=train.iloc[:,[2,4,5,6,7,10,11,12]]

x_train.head(2)
y_train=train.iloc[:,[1]]

y_train.head(3)
x_test=test.iloc[:,[1,3,4,5,6,9,10,11]]

x_test.head(3)
from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.metrics import classification_report
# Logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver="lbfgs",max_iter=10000)

logreg.fit(x_train, y_train.values.ravel())

y_predlog=logreg.predict(x_train)

print("Confusion Matrix")

print(DataFrame(confusion_matrix(y_train,y_predlog)))

print("Accuracy Score",accuracy_score(y_train,y_predlog))

from sklearn import svm
# Sigmoid kernel

clf_s=svm.SVC(kernel="sigmoid",C=3.6,gamma=0.36)

clf_s.fit(x_train,y_train.values.ravel())

y_predtr=clf_s.predict(x_train)

print("Confusion Matrix")

print(DataFrame(confusion_matrix(y_train,y_predtr)))

print("Accuracy Score",accuracy_score(y_train,y_predtr))
# Linear kernel

clf_l=svm.SVC(kernel="linear")

clf_l.fit(x_train,y_train.values.ravel())

y_predl=clf_l.predict(x_train)

print("Confusion Matrix")

print(DataFrame(confusion_matrix(y_train,y_predl)))

print("Accuracy Score",accuracy_score(y_train,y_predl))
# Polynomial kernel

clf_p=svm.SVC(kernel="poly",C=3.6,gamma=0.36,degree=2)

clf_p.fit(x_train,y_train.values.ravel())

y_predp=clf_p.predict(x_train)

print("Confusion Matrix")

print(DataFrame(confusion_matrix(y_train,y_predp)))

print("Accuracy Score",accuracy_score(y_train,y_predp))
# Radial Basis Function kernel

clf_r=svm.SVC(kernel="rbf",C=3.6,gamma=0.36)

clf_r.fit(x_train,y_train.values.ravel())

y_predr=clf_r.predict(x_train)

print("Confusion Matrix")

print(DataFrame(confusion_matrix(y_train,y_predr)))

print("Accuracy Score",accuracy_score(y_train,y_predr))
# Predict final 

test["Survived"]=clf_r.predict(x_test)
# Top Head after prediction

test.head(6)
#Final Submission

sub=test.iloc[:,[0,12]]

sub.head()
# Export Submission

sub.to_csv("Titanic_survival.csv",index=False)