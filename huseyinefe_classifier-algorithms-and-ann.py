

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visulization

import seaborn as sns # data visulization

from copy import deepcopy # data copy

from sklearn.model_selection import train_test_split # data split

from sklearn.preprocessing import StandardScaler # data preprocessing

from sklearn.ensemble import RandomForestClassifier # machine learning algorithm

from sklearn.svm import SVC # machine learning algorithm

from sklearn.neighbors import KNeighborsClassifier # machine learning algorithm

from sklearn.linear_model import LogisticRegression # machine learning algorithm

from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score # measure error algorithm

from keras.models import Sequential # start ANN model Deep Learning tool

from keras.layers import Dense,Dropout # Deep Learning tool for prepare ANN model 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset=pd.read_csv("../input/heart-disease-uci/heart.csv")
dataset.info()
dataset.head()
plt.figure(figsize=(10,10))

sns.countplot(dataset.age[dataset.target == 1 ])

plt.title("Age Distribution of People That Being Heart Diases")

plt.ylabel("Number of People")

plt.xlabel("Age of People")

plt.show()
plt.figure(figsize=(10,10))

sns.countplot(dataset.age[dataset.target == 0 ])

plt.title("Age Distribution of People That Not Being Heart Diases")

plt.ylabel("Number of People")

plt.xlabel("Age of People")

plt.show()
x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1:].values

print("x shape :",x.shape)

print("y shape :",y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

print("x_train shape :",x_train.shape)

print("x_test shape :",x_test.shape)

print("y_train shape :",y_train.shape)

print("y_test shape :",y_test.shape)
sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
model = Sequential()



model.add(Dense(206,activation="relu",input_dim=13))

model.add(Dropout(0.4))

model.add(Dense(103,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
hist=model.fit(x_train,y_train,batch_size=250,epochs=100,validation_data=(x_test,y_test))
prediction=model.predict(x_test)

predict=deepcopy(prediction)

for i in range(0,len(prediction)):

    if prediction[i] <0.5:

        prediction[i] = 0

    else:

        prediction[i] = 1
cfm=confusion_matrix(y_test,prediction)

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)

plt.title("Error Number With Heat Map")

plt.xlabel("Real")

plt.ylabel("Prediction")

plt.show()
fpr,tpr,threshold=roc_curve(y_test,predict)

print("fpr shape :",fpr.shape)

print("tpr shape :",tpr.shape)

print("threshold shape :",threshold.shape)
plt.figure(figsize=(15,8))

plt.plot(tpr,color="green",label="TPR")

plt.plot(fpr,color="red",label="FPR")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.grid()

plt.title("Roc Curve Validation")

plt.show()
score=roc_auc_score(y_test,predict)

print("Roc Auc Score :",score)
train_score=[]

test_score=[]

for i in range(1,11):

 rfc=RandomForestClassifier(n_estimators=i,random_state=0)

 rfc.fit(x_train,y_train)

 train_score.append(rfc.score(x_train,y_train))

 test_score.append(rfc.score(x_test,y_test))
plt.plot(train_score,color="green",label="Train Score")

plt.plot(test_score,color="red",label="Test Score")

plt.legend()

plt.xlabel("Number of Tree")

plt.ylabel("Score Validations")

plt.title("Choose Tree Number")

plt.show()
rfc1=RandomForestClassifier(n_estimators=3,random_state=0)

rfc1.fit(x_train,y_train)

prediction=rfc1.predict(x_test)

print("Train Accuracy :",rfc1.score(x_train,y_train))

print("Test Accuracy :",rfc1.score(x_test,y_test))
f,ax=plt.subplots(figsize=(10,10))

cfm=confusion_matrix(y_test,prediction)

sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)

plt.title("Error Number With Heat Map")

plt.xlabel("Real")

plt.ylabel("Prediction")

plt.show()
svc=SVC(random_state=42)

svc.fit(x_train,y_train)

print("Train Accuracy :",svc.score(x_train,y_train))

print("Test Accuracy :",svc.score(x_test,y_test))
prediction=svc.predict(x_test)
f,ax=plt.subplots(figsize=(10,10))

cfm=confusion_matrix(y_test,prediction)

sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)

plt.title("Error Number With Heat Map")

plt.xlabel("Real")

plt.ylabel("Prediction")

plt.show()
train_score=[]

test_score=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    train_score.append(knn.score(x_train,y_train))

    test_score.append(knn.score(x_test,y_test))
plt.plot(train_score,color="green",label="Train Score")

plt.plot(test_score,color="red",label="Test Score")

plt.legend()

plt.xlabel("Number of Neighbours")

plt.ylabel("Score Validations")

plt.title("Choose Tree Number")

plt.show()
knn1=KNeighborsClassifier(n_neighbors=6)

knn1.fit(x_train,y_train)

print("Train Accuracy :",knn1.score(x_train,y_train))

print("Test Accuracy :",knn1.score(x_test,y_test))
prediction=knn1.predict(x_test)
f,ax=plt.subplots(figsize=(10,10))

cfm=confusion_matrix(y_test,prediction)

sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)

plt.title("Error Number With Heat Map")

plt.xlabel("Real")

plt.ylabel("Prediction")

plt.show()
lr=LogisticRegression()

lr.fit(x_train,y_train)

print("Train Accuracy :",lr.score(x_train,y_train))

print("Test Accuracy :",lr.score(x_test,y_test))
prediction=lr.predict(x_test)
f,ax=plt.subplots(figsize=(10,10))

cfm=confusion_matrix(y_test,prediction)

sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)

plt.title("Error Number With Heat Map")

plt.xlabel("Real")

plt.ylabel("Prediction")

plt.show()