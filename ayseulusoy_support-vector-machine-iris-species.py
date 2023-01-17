import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sea 

from sklearn import svm

from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)
iris.feature_names
df=pd.DataFrame(iris.data,columns=iris.feature_names)

df.head()
iris.target_names
df['target']=iris.target

df.head()
df[df.target==1].head()
df[df.target==2].head()
df.tail()
df['flower_name']=df.target.apply(lambda x:iris.target_names[x])

df.head()
df.tail()
df0=df[df.target==0]

df1=df[df.target==1]

df2=df[df.target==2]
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='red',marker='*')

plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='*')

plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='green',marker='*')

#we can't apply svm between df1 and df2 therefore we'll apply svm between df0 and df1.
plt.xlabel('sepal length (cm)')

plt.ylabel('sepal width (cm)')

plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='red',marker='*')

plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='*')
plt.xlabel('petal length (cm)')

plt.ylabel('petal width (cm)')

plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='red',marker='*')

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='*')
from sklearn.model_selection import train_test_split #we're splitting the data set  into training and test set
X=df.drop(['target','flower_name'],axis='columns')

X.head()
Y=df.target
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
len(X_train)
len(X_test)
svc=svm.SVC(gamma='auto')
svc.fit(X_train,Y_train)
svc.score(X_test,Y_test)