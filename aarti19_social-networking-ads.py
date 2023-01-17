#importing important libraries

import pandas as pd  #for data analysis

import numpy as np   #for linear algebra

import matplotlib.pyplot as plt   #for visulaization

import seaborn as sns

%matplotlib inline
#importing dataset

data=pd.read_csv('../input/-social-networking-ads/Social_Network_Ads.csv')
# displaying all the features of the data

data.columns
data.head()
#Checking the shape of the dataset

data.shape 
#checking the information of the Data

data.info()
#Checking if there is any missing Value present in the dataset

data.isnull().sum()

#There is no missing value present in the dataset
#Checking the Datatype of the features

data.dtypes
#Information About the Data

data.describe()
data['Purchased'].value_counts()
data['Purchased'].value_counts().plot.bar()
sns.distplot(data['Age'])
sns.distplot(data['EstimatedSalary'])

#EstimatedSalary is normally distributed
plt.boxplot(data['EstimatedSalary'])

#Boxplot help us to check if there is any outlier present in the feature

#There is no outlier present in the EstimatedSalary Feature
plt.boxplot(data['Age'])

#There is no outlier present in the Age feature as well
plt.boxplot(data['User ID'])
#Scatter Plot of estimatedsalary and Dependent variable purchased

sns.scatterplot(x='EstimatedSalary',y='Purchased',data=data)
sns.scatterplot(x='Age',y='Purchased',data=data)
sns.scatterplot(x='User ID',y='Purchased',data=data)
#We have one Categorical Variable as well

data.Gender.unique()
df1=pd.get_dummies(data=data)

#Here we change the categorical variable into continous variable
#Here we change the categorical variable Gender into continous variables Gender_Male and Gender_female

df1.head()
#Correlation between different features of our dataset

sns.heatmap(df1.corr(),annot=True)
#The dependent variable Purchased is very less correlated with User ID so we drop that feature

#We creates dummy variables so we drop one 

df2=df1.drop('User ID',axis=1)

df2=df2.drop('Gender_Male',axis=1)
df2.head()
#Separating Depenent and Independent Features

X=df2.iloc[:,[0,1,3]]

Y=df2.iloc[:,2]
X.head()
Y.head()
#Checking the shape of independent and dependent variables

print("Shape of Independent features:",X.shape)

print("Shape of dependent feature:",Y.shape)
plt.title("Correlation matrix")

sns.heatmap(df2.corr(),annot=True)

#Splitting the Data into training and test dataset

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print("Shape of X_train:",X_train.shape)

print("Shape pf X_test:",X_test.shape)

print("shape of Y_train:",Y_train.shape)

print("Shape of Y_test:",Y_test.shape)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

XX_train=sc.fit_transform(X_train)

XX_test=sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

loreg=LogisticRegression()
#Fitting logistic Regression to the training set

loreg.fit(XX_train,Y_train)
#Score of the training set

loreg.score(XX_train,Y_train)
#Predicting the test Dataset

pred=loreg.predict(XX_test)
# Accuracy Score of test set

from sklearn.metrics import accuracy_score

accuracy_score(Y_test,pred)
inter=loreg.intercept_

print(inter)
#Coefficients of regression model

coeff=loreg.coef_

print(coeff)
#Making Confusion Matrix

from sklearn import metrics

cm=metrics.confusion_matrix(Y_test,pred)

print(cm)
TP=48

FP=12

TN=20

FN=3

acc=(TP+TN)/(TP+TN+FP+FN)

rc=TP/(TP+FN)

pre=TP/(TP+FP)
#Printing Accuracy,Recall and Precision



print(acc)

print(rc)

print(pre)
f_measure=(2*rc*pre)/(rc+pre)

print(f_measure)
#Importing support vector classifier

from sklearn.svm import SVC

svc=SVC()
#fitting the training set 

svc.fit(XX_train,Y_train)
#predicting the test dataset

pred1=svc.predict(XX_test)
#accuracy of training dataset

svc.score(XX_train,Y_train)
from sklearn.metrics import accuracy_score

accuracy_score(Y_test,pred1)
from sklearn import metrics
#Making Confusion matrix

cm1=metrics.confusion_matrix(Y_test,pred1)

print(cm1)
TP=45

FP=4

FN=3

TN=28

acc=(TP+TN)/(TP+TN+FP+FN)

rc=TP/(TP+FN)

pre=TP/(TP+FP)

print(acc)

print(rc)

print(pre)