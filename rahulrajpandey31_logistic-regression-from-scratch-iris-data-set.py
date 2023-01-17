#importing libraries 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
df= pd.read_csv("/kaggle/input/iris/Iris.csv")   #importing dataset and making dataframe 

df.head()                                        #showing top 5 data entry 
df.describe() #describes are data 
df.info() #gives information about the columns
df.shape #tells us about no. of rows and column [rows , columns]
df.drop("Id",axis=1,inplace=True)    #droping  id becuase it is no use to us , Inplace = True means changes will take effect in original dataframe

df.head()
print(df["Species"].value_counts())

sns.countplot(df["Species"])
plt.figure(figsize=(8,4)) 

sns.heatmap(df.corr(),annot=True,fmt=".0%") #draws  heatmap with input as the correlation matrix calculted by(df.corr())

plt.show()

# We'll use seaborn's FacetGrid to color the scatterplot by species

sns.FacetGrid(df, hue="Species", height=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
#let Create a pair plot of some columns 

sns.pairplot(df.iloc[:,:],hue='Species')  # graph also  tell us about the the realationship between the two columns 
# We can quickly make a boxplot with Pandas on each feature split out by species

df.boxplot(by="Species", figsize=(15,15))
# importing alll the necessary packages to Logistic Regression 

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn import metrics #for checking the model accuracy
X=df.iloc[:,0:4]

Y=df["Species"]

X.head()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)# in this our main data is split into train and test

# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%

print("Train Shape",X_train.shape)

print("Test Shape",X_test.shape)
log = LogisticRegression()

log.fit(X_train,Y_train)

prediction=log.predict(X_test)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,Y_test))
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn.svm import SVC   #for Support Vector Machine (SVM) Algorithm

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
tree=DecisionTreeClassifier()

tree.fit(X_train,Y_train)

prediction=tree.predict(X_test)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,Y_test))
knn=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

knn.fit(X_train,Y_train)

prediction=knn.predict(X_test)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,Y_test))
svc=SVC()

svc.fit(X_train,Y_train) 

prediction=svc.predict(X_test)

print('The accuracy of the SVC is',metrics.accuracy_score(prediction,Y_test))
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)

forest.fit(X_train,Y_train)

print('The accuracy of the SVC is',metrics.accuracy_score(prediction,Y_test))