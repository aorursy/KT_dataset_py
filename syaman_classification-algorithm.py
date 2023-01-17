import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#we will import our data
mydata=pd.read_csv("../input/column_2C_weka.csv")

#in our data we have 2 kind class which are Abnormal and Normal
mydata["class"].unique()

#now we will seperate our data in two parth, 
Abnormal=mydata[mydata["class"]=="Abnormal"]
Normal=mydata[mydata["class"]=="Normal"]

#visalize our data
plt.scatter(Abnormal.pelvic_incidence,Abnormal.sacral_slope,color="red",label="Abnormal",alpha=0.4)
plt.scatter(Normal.pelvic_incidence,Normal.sacral_slope,color="green",label="Normal",alpha=0.6)
plt.legend()#this command will show labels
plt.xlabel("pelvic_incidence")
plt.ylabel("pelvic_radius")
plt.show()

#now we will seperate our data in two part
mydata["class"]= [1 if each == "Abnormal" else 0 for each in mydata["class"]]

y=mydata["class"].values
x=mydata.drop(["class"],axis=1)

#normalization (x-max(x))/(max(x)-min(x))
x_normal = (x-np.max(x))/(np.max(x)-np.min(x))

#train test split
xtrain,xtest,ytrain,ytest=train_test_split(x_normal,y,test_size=0.3,random_state=21)

#SVM Model
svm=SVC(random_state=1)
svm.fit(xtrain,ytrain)

print("accuracy of svm algoritm:",svm.score(xtest,ytest))

#Naive Bayes
nb=GaussianNB()
nb.fit(xtrain,ytrain)
print("accuracy of naive bayes algorithm",nb.score(xtest,ytest))
#Decision Tree
# we made normalization for knn, and we will use it here too
xtrain,xtest,ytrain,ytest=train_test_split(x_normal,y,test_size=0.15,random_state=21)

dt=DecisionTreeClassifier()
dt.fit(xtrain,ytrain)

#prediction
print("accuracy of decision tree algorithm",dt.score(xtest,ytest))
#Random Forest 
rf=RandomForestClassifier(n_estimators=100,random_state=21) #n_estimators=number of trees
rf.fit(xtrain,ytrain)
print("random forest accuracy :",rf.score(xtest,ytest))
#confusion matrix (member of model evaluation matrix)
#we will see our data accuracy and fail rate for each class which are Abnormal and Normal
#first, we need prediction
y_prediction=rf.predict(xtest)
#later we need true datas
y_true=ytest


cm=confusion_matrix(y_true,y_prediction)

#confusion matrix visualization

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y prediction")
plt.ylabel("y true")
plt.show()

#Confusion matrix with random forest
x,y=mydata.loc[:,mydata.columns !="class"],mydata.loc[:,"class"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)

rf=RandomForestClassifier(random_state=4)
rf.fit(xtrain,ytrain)
y_prediction=rf.predict(xtest)

conm=confusion_matrix(ytest,y_prediction)
print("confusion matrix:\n",conm)
print("Classification report: \n",classification_report(ytest,y_prediction))
