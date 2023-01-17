#Importing all required libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
#Reading file

df=pd.read_csv("../input/heart-disease-uci/heart.csv")

df.head()
df.describe
#check for null values

df.isnull().sum()
pd.crosstab(df.age,df.target).plot(kind='bar',figsize=(15,6))

plt.show()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
#seperating target and features

y=df['target']

y

x=df.drop(['target'],axis=1)

x.head()
#Slitting training and testing data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state = 42)

m1=LogisticRegression(solver='liblinear', multi_class='ovr')

m2=LinearDiscriminantAnalysis()

m3=KNeighborsClassifier()

m4=DecisionTreeClassifier()

m5=GaussianNB()

m6=SVC(gamma='auto')

models = []

models.append(('LR', m1))

models.append(('LDA', m2))

models.append(('KNN', m3))

models.append(('CART', m4))

models.append(('NB', m5))

models.append(('SVM', m6))

# evaluate each model in turn

results = []

names = []

for name,m in models:

  m.fit(x_train,y_train)

  pred3=m.predict(x_test)

  sc=accuracy_score(y_test,pred3)

  results.append(sc)

  names.append(name)

  print( (name, sc))

#Performing Normalization on data 

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(x)

x=scaler.transform(x)

x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state = 42)

m1=LogisticRegression(solver='liblinear', multi_class='ovr')

m2=LinearDiscriminantAnalysis()

m3=KNeighborsClassifier()

m4=DecisionTreeClassifier()

m5=GaussianNB()

m6=SVC(gamma='auto')

models = []

models.append(('LR', m1))

models.append(('LDA', m2))

models.append(('KNN', m3))

models.append(('CART', m4))

models.append(('NB', m5))

models.append(('SVM', m6))

# evaluate each model in turn

results = []

names = []

for name,m in models:

  m.fit(x_train,y_train)

  pred3=m.predict(x_test)

  sc=accuracy_score(y_test,pred3)

  results.append(sc)

  names.append(name)

  print( (name, sc))

#Performing standerdization

from sklearn.preprocessing import StandardScaler

s=StandardScaler()

s.fit(x)

x=s.transform(x)

x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state = 42)

m1=LogisticRegression(solver='liblinear', multi_class='ovr')

m2=LinearDiscriminantAnalysis()

m3=KNeighborsClassifier()

m4=DecisionTreeClassifier()

m5=GaussianNB()

m6=SVC(gamma='auto')

models = []

models.append(('LR', m1))

models.append(('LDA', m2))

models.append(('KNN', m3))

models.append(('CART', m4))

models.append(('NB', m5))

models.append(('SVM', m6))

# evaluate each model in turn

results = []

names = []

for name,m in models:

  m.fit(x_train,y_train)

  pred3=m.predict(x_test)

  sc=accuracy_score(y_test,pred3)

  results.append(sc)

  names.append(name)

  print( (name, sc))
