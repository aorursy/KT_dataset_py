import pandas as pd

import numpy as np

# importing plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

# Splits the dataframe into random train and test subsets to to 

# train the model and test

from sklearn.model_selection import train_test_split

# Gaussian Naive bayes for model building

from sklearn.naive_bayes import GaussianNB

# Metrics for accuracy and confusion matrix and classification report

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler, StandardScaler,scale
# To enable plotting graphs

%matplotlib inline 
# Read the data from CSV using pandas 

df = pd.read_csv('../input/pima-indians-diabetes_nb.csv')
# Check data is read into pandas dataframe

df.head()
# Check for number rows and columns present in the dataset

df.shape
# To check data type of columns in the dataframe

df.info()
# Check for null values

df.isnull().any()
df[~df.applymap(np.isreal).all(1)]
# Check target column 'class' count 

df['class'].value_counts()
# See the plot for categorical target column count

sns.countplot(df['class'])
# Distribution of the attributes in the dataset

df.describe().T
# See the histograms for each independant variable present in dataframe df
# min is zero and right skewed

df['Preg'].hist()
# Left skewed

df['Plas'].hist()
# left skewed and outliers present

df['Pres'].hist()
# min zero more right skewed

df['skin'].hist()
# Large right skewed

df['test'].hist()
# Mean and median almost near 

df['mass'].hist()
# mean and median little near

df['pedi'].hist()
# data points largely right skewed 

sns.distplot(df['age'])
df['class'].hist()
sns.pairplot(df, hue = 'class')
# When observe the pairplot plots and distribution of data (Plas,Pres, skin and mass) are nearly normally distributed

# Attributes (Preg, test, pedi ) has exponential distribution
sns.boxplot(x = df['Preg'] )
sns.boxplot(x = df['test'] )
df.isnull().any()
corr = df.corr()
sns.heatmap(corr,annot = True)
df.groupby('class').hist(figsize=(9, 9))
df1 = df
# There are few zero values as this attribute is contnuous consider as missing values

#df1['Plas'] = df1['Plas'].replace(0,np.NaN)
#df1['Plas'] = df1['Plas'].replace(np.NaN, df1['Plas'].mean(skipna=True))
df1.head()
# drop target value for X values independent attributes

X = df1.drop(['class'], axis =1)

# target attribute - dependant attribute

y = df1['class']
# split the data 30 %

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
# Gaussian Naivebayes model

gnb = GaussianNB()
# train the model with train split

gnb.fit(X_train,y_train)
# Check the score for train data

gnb.score(X_train,y_train)
# Predict the target value for test data

y_predict = gnb.predict(X_test)
# Check the score for test data 

gnb.score(X_test,y_test)
# see the confusion matrics for precision and recall

print(metrics.confusion_matrix(y_test,y_predict))
# see the f1 score and precision , recall percentage

print(metrics.classification_report(y_test,y_predict))
print(metrics.accuracy_score(y_test,y_predict))
X = df1.drop(['class'], axis =1)
y = df1['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
from sklearn.preprocessing import scale

X_train_scale = scale(X_train)

X_test_scale = scale(X_test)
X_test_scale.shape
X_train_scale.shape
X_train_scale
gnb = GaussianNB()
gnb.fit(X_train_scale,y_train)
gnb.score(X_train_scale,y_train)
y_predict = gnb.predict(X_test_scale)
gnb.score(X_test_scale,y_test)
print(metrics.classification_report(y_test,y_predict))
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.accuracy_score(y_test,y_predict))
X = df1.drop(['class'], axis =1)
y = df1['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)

X_test_scale = scale.fit_transform(X_test)
X_test_scale.shape
X_train_scale.shape
X_train_scale
gnb = GaussianNB()
gnb.fit(X_train_scale,y_train)
gnb.score(X_train_scale,y_train)
y_predict = gnb.predict(X_test_scale)
gnb.score(X_test_scale,y_test)
print(metrics.classification_report(y_test,y_predict))
print(metrics.confusion_matrix(y_test,y_predict))
print(metrics.accuracy_score(y_test,y_predict))