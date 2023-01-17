import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report,confusion_matrix
df= pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')
df.head()
# this to show the data type , columns names ,sum of data that not a nans and memory usage . 

df.info()
# show the number of missing values in each column.

df.isna().sum().sort_values(ascending=False)
# Removing some columns that have > 28% nan values

max_nans=len(df)*0.28

df=df.loc[:, (df.isnull().sum(axis=0) <= max_nans)]
# show the number of the missing values 

df.isna().sum().sort_values(ascending=False)
#filling the missing values by the next value ('bfill') because the temperatures are nearly the same for the next day

df.fillna(method='bfill',inplace=True)
df.drop(columns=['RISK_MM','Date','Location'],inplace=True)
# here we change the data type for these two column

df['RainTomorrow']=df['RainTomorrow'].astype('category')

df['RainToday']=df['RainToday'].astype('category')
# here we transform them to numeric values.

df['RainTomorrow']=df['RainTomorrow'].cat.codes

df['RainToday']=df['RainToday'].cat.codes
 # to know the number of row and columns .

df.shape
sns.countplot(df['RainToday'])
sns.countplot(df['RainTomorrow'])
g = sns.FacetGrid(df,hue='RainTomorrow',aspect=4)

g.map(plt.hist,'MinTemp',alpha=0.6,bins=5)

plt.legend()
g = sns.FacetGrid(df,hue='RainTomorrow',height=6,aspect=2)

g.map(plt.hist,'Humidity3pm',alpha=0.6,bins=5)

plt.legend()
g = sns.FacetGrid(df,hue='RainTomorrow',height=6,aspect=2)

g.map(plt.hist,'WindGustSpeed',alpha=0.6,bins=5)

plt.legend()
#select the columns for the model.

X=df.iloc[:,:16]
X.head()
# select our target .

y=df['RainTomorrow']
# here we transform the non_numeric features to make the model dealing with it . 

X=pd.get_dummies(X,columns=['WindDir9am','WindDir3pm','WindGustDir'],drop_first=True)
#Next, we split 75% of the data to the training set while 25% of the data to test set using below code.

X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0,stratify=y)
#we need to bring all features to the same level of magnitudes. This can be achieved by a method called feature scaling.

scaler = preprocessing.MinMaxScaler()

scaler.fit(X)

X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

X.iloc[4:10]
#Our next step is to K-NN model and train it with the training data. Here n_neighbors is the value of factor K.

classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(X_train,y_train)
# Let's train our test data and check its accuracy.

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)
# let's see the classification report .

y_test_pred = classifier.predict(X_test)

print(classification_report(y_test,y_test_pred))
#let's show the confusion matrix.

confusion_matrix(y_test,y_test_pred)
classifier = KNeighborsClassifier(n_neighbors = 8, p = 2)

classifier.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

y_test_pred = classifier.predict(X_test)

print(classification_report(y_test,y_test_pred))