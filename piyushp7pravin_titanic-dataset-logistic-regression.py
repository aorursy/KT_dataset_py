import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
training_data = pd.read_csv('/kaggle/input/titanic/train.csv')
print(training_data.shape)
training_data.head(10)
training_data.shape
training_data.describe(include="all")
percent_NaN = (training_data.isnull().sum()/training_data.shape[0])*100
print(percent_NaN)
training_data["Age"].hist(bins=10, stacked=True,density=True,color="orange",alpha=0.8, width=7)
training_data["Age"].plot(kind='density', color='blue')
plt.xlim(-10,100)
plt.xlabel("AGE")
plt.show()
median = training_data["Age"].median(skipna=True)
training_data["Age"].fillna(median,inplace=True)
embarked_value = training_data["Embarked"].value_counts().index
embarked_count = training_data["Embarked"].value_counts().values

barlist = plt.bar(embarked_value,embarked_count,width=0.5,alpha=0.8)
barlist[0].set_color('orange')
barlist[1].set_color('teal')
barlist[2].set_color('blue')
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.show()
training_data["Embarked"].fillna("S",inplace=True)
training_data["Embarked_S"]=training_data["Embarked"]=="S"
training_data["Embarked_C"]=training_data["Embarked"]=="C"
training_data["Embarked_Q"]=training_data["Embarked"]=="Q"

training_data.loc[training_data["Embarked_S"]==True,"Embarked_S"] = 1
training_data.loc[training_data["Embarked_S"]==False,"Embarked_S"] = 0

training_data.loc[training_data["Embarked_C"]==True,"Embarked_C"] = 1
training_data.loc[training_data["Embarked_C"]==False,"Embarked_C"] = 0

training_data.loc[training_data["Embarked_Q"]==True,"Embarked_Q"] = 1
training_data.loc[training_data["Embarked_Q"]==False,"Embarked_Q"] = 0
training_data.head()
training_data.loc[training_data["Sex"]=="male","Sex"] = 1
training_data.loc[training_data["Sex"]=="female","Sex"] = 0
title_training=[]
for i in training_data['Name']:
    title_training.append(i.split(',')[1].split('.')[0].strip())
title_training=np.array(title_training)
title_training[title_training=='Master']=0
title_training[title_training=='Miss']=1
title_training[title_training=='Mr']=2
title_training[title_training=='Mrs']=3
title_training[(title_training!='0')&(title_training!='1')&(title_training!='2')&(title_training!='3')]=4
title_training=np.array(title_training, dtype='int')
training_data.drop("Name",axis=1,inplace=True)
training_data.drop("Cabin",axis=1,inplace=True)
training_data.drop("Ticket",axis=1,inplace=True)
training_data.drop("Embarked",axis=1,inplace=True)
training_data.head()
training_data["Is_Minor"]=training_data["Age"]<18
training_data.loc[training_data["Is_Minor"]==True,"Is_Minor"] = 1
training_data.loc[training_data["Is_Minor"]==False,"Is_Minor"] = 0
training_data.head()
training_data["Survived_"]=training_data["Survived"]
training_data.drop("Survived",axis=1,inplace=True)
training_data.head()
X_train = training_data.values[:,:-1]
X_train=np.append(X_train, title_training.reshape(-1, 1), axis=1)
Y_train = training_data.values[:,-1]
square = []
for i in X_train:
    square.append(i**2)
square = np.array(square)
X_train = np.append(X_train, square, axis = 1)
travel = []
travel.append(np.where((training_data["SibSp"]+training_data["Parch"])>0, 0, 1))

travel = np.array(travel)
X_train = np.append(X_train, travel.reshape(-1,1), axis = 1)
age_fare = []
for i in training_data["Age"]*training_data["Fare"]:
    age_fare.append(i)
age_fare = np.array(age_fare)
X_train = np.append(X_train, age_fare.reshape(-1,1), axis = 1)
age_add_fare = []
for i in training_data["Age"]+training_data["Fare"]:
    age_add_fare.append(i)
age_add_fare = np.array(age_add_fare)
X_train = np.append(X_train, age_add_fare.reshape(-1,1), axis = 1)
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
print(X_train.shape,Y_train.shape)
testing_data = pd.read_csv('/kaggle/input/titanic/train.csv')
print(testing_data.shape)
testing_data.head(10)
testing_data.shape
testing_data.describe(include="all")
percent_NaN = (testing_data.isnull().sum()/testing_data.shape[0])*100
print(percent_NaN)
testing_data["Age"].hist(bins=10, stacked=True,density=True,color="orange",alpha=0.8, width=6)
testing_data["Age"].plot(kind='density', color='blue')
plt.xlim(-10,100)
plt.xlabel("AGE")
plt.show()
median = testing_data["Age"].median(skipna=True)
testing_data["Age"].fillna(median,inplace=True)
embarked_value = testing_data["Embarked"].value_counts().index
embarked_count = testing_data["Embarked"].value_counts().values

barlist = plt.bar(embarked_value,embarked_count,width=0.5,alpha=0.8)
barlist[0].set_color('orange')
barlist[1].set_color('teal')
barlist[2].set_color('blue')
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.show()
testing_data["Embarked"].fillna("S",inplace=True)
testing_data["Embarked_S"]=testing_data["Embarked"]=="S"
testing_data["Embarked_C"]=testing_data["Embarked"]=="C"
testing_data["Embarked_Q"]=testing_data["Embarked"]=="Q"

testing_data.loc[testing_data["Embarked_S"]==True,"Embarked_S"] = 1
testing_data.loc[testing_data["Embarked_S"]==False,"Embarked_S"] = 0

testing_data.loc[testing_data["Embarked_C"]==True,"Embarked_C"] = 1
testing_data.loc[testing_data["Embarked_C"]==False,"Embarked_C"] = 0

testing_data.loc[testing_data["Embarked_Q"]==True,"Embarked_Q"] = 1
testing_data.loc[testing_data["Embarked_Q"]==False,"Embarked_Q"] = 0
testing_data.head()
testing_data.loc[testing_data["Sex"]=="male","Sex"] = 1
testing_data.loc[testing_data["Sex"]=="female","Sex"] = 0
title_testing=[]
for i in testing_data['Name']:
    title_testing.append(i.split(',')[1].split('.')[0].strip())
title_testing=np.array(title_testing)
title_testing[title_testing=='Master']=0
title_testing[title_testing=='Miss']=1
title_testing[title_testing=='Mr']=2
title_testing[title_testing=='Mrs']=3
title_testing[(title_testing!='0')&(title_testing!='1')&(title_testing!='2')&(title_testing!='3')]=4
title_testing=np.array(title_testing, dtype='int')
testing_data.drop("Name",axis=1,inplace=True)
testing_data.drop("Cabin",axis=1,inplace=True)
testing_data.drop("Embarked",axis=1,inplace=True)
testing_data.drop("Ticket",axis=1,inplace=True)
testing_data.head()
testing_data["Is_Minor"]=testing_data["Age"]<18
testing_data.loc[testing_data["Is_Minor"]==True,"Is_Minor"] = 1
testing_data.loc[testing_data["Is_Minor"]==False,"Is_Minor"] = 0
testing_data.head()
X_test = testing_data.values
X_test=np.append(X_test, title_testing.reshape(-1, 1), axis=1)
square = []
for i in X_test:
    square.append(i**2)
square = np.array(square)
X_test = np.append(X_test, square, axis = 1)
travel = []
travel.append(np.where((testing_data["SibSp"]+testing_data["Parch"])>0, 0, 1))

travel = np.array(travel)
X_test = np.append(X_test, travel.reshape(-1,1), axis = 1)
age_fare = []
for i in testing_data["Age"]*testing_data["Fare"]:
    age_fare.append(i)
age_fare = np.array(age_fare)
X_test = np.append(X_test, age_fare.reshape(-1,1), axis = 1)
age_add_fare = []
for i in testing_data["Age"]+testing_data["Fare"]:
    age_add_fare.append(i)
age_add_fare = np.array(age_add_fare)
X_test = np.append(X_test, age_add_fare.reshape(-1,1), axis = 1)
scaler.fit(X_test)
X_test = scaler.transform(X_test)
clf=LogisticRegression(C=2,solver="saga",max_iter=10000000,tol=0.00001)
clf
clf.fit(X_train, Y_train)
clf.score(X_train, Y_train)
Y_pred=clf.predict(X_test)
Y_pred
