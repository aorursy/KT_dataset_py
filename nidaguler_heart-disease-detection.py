import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/heart.csv")

df.head()
#Count the number of rows and columns in the daha set

df.shape
#count the number of missing values in each columns

df.isna().sum()
#get a count of the number of target(1) or not(0)

df.target.value_counts()
#visualize the count

sns.countplot(df.target,label="count")

plt.show()
#create a pair plot

sns.pairplot(df,hue="target")

plt.show()
df.corr()
#visualize the correlation

plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), annot=True,fmt=".0%")

plt.show()
#Split the data set into independent(x) and dependent (y) data sets

x=df.iloc[:,0:13].values

y=df.iloc[:,-1].values
#split the data set into 75% training and 25% testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#scale the data(feature scaling)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
#create a function for the models

def models(x_train,y_train):

  #Logistic Regression Model

  from sklearn.linear_model import LogisticRegression

  log=LogisticRegression(random_state=0)

  log.fit(x_train,y_train)

  

  #Decision Tree

  from sklearn.tree import DecisionTreeClassifier

  tree=DecisionTreeClassifier(criterion='entropy',random_state=0)

  tree.fit(x_train,y_train)

  

  #Random Forest Classifier

  from sklearn.ensemble import RandomForestClassifier

  forest = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)

  forest.fit(x_train,y_train)



  #Print the models accuracy on the training data

  print("[0]Logistic Regression Training Accuracy:",log.score(x_train,y_train))

  print("[1]Decision Tree Classifier Training Accuracy:",tree.score(x_train,y_train))

  print("[2]Random Forest Classifier Training Accuracy:",forest.score(x_train,y_train))

  

  return log,tree,forest
#Getting all of the models

model = models(x_train,y_train)
#test model accuracy on confusion matrix

from sklearn.metrics import confusion_matrix



for i in range(len(model)):

  print("Model ", i)

  cm =confusion_matrix(y_test,model[i].predict(x_test))



  TP=cm[0][0]

  TN=cm[1][1]

  FN=cm[1][0]

  FP=cm[0][1]



  print(cm)

  print("Testing Accuracy = ", (TP+TN) / (TP+TN+FN+FP))

  print()
#show another way to get metrics of the models

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



for i in range(len(model) ):

  print("Model ",i)

  print( classification_report(y_test,model[i].predict(x_test)))

  print( accuracy_score(y_test,model[i].predict(x_test)))

  print()
#print the prediction of random forest classifier model

pred=model[2].predict(x_test)

print(pred)

print()

print(y_test)