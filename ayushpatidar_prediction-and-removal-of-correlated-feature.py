import pandas as pd
import numpy as np
import seaborn as  sns
from matplotlib import pyplot as py
from sklearn.metrics import confusion_matrix
%matplotlib inline
train=pd.read_csv("../input/data.csv")#reading the data through pandas 
train.head()
train.shape
train.isnull().sum()
train.drop("Unnamed: 32",axis=1,inplace=True)#removing unnamed feature because it has all values as null
f,ax=py.subplots(figsize=(23,20))
cormat=train.corr()
sns.heatmap(cormat,annot=True)#creating heatmap to undersatnd relation between features


#train["hi"]=pd.cut(train["radius_mean"],5)

f,ax=py.subplots(figsize=(20,20))
py.subplot(2,2,1)
sns.swarmplot(x="diagnosis",y="texture_mean",data=train)
py.subplot(2,2,2)
sns.swarmplot(x="diagnosis",y="radius_mean",data=train)
py.subplot(2,2,3)
sns.swarmplot(x="diagnosis",y="area_mean",data=train)
py.subplot(2,2,4)
sns.swarmplot(x="diagnosis",y="perimeter_mean",data=train)

train.drop(["radius_worst","perimeter_worst","area_worst"],axis=1,inplace=True)#dropping features because the have a great coorelation in betwwen and retaining only perimetre_worst
sns.swarmplot(x="diagnosis",y="symmetry_mean",data=train)#as we symmetry_mean is not useful in classifying whether cancer is "M" oR "B" so it is not useful
train.drop(["symmetry_mean","symmetry_se","symmetry_worst"],axis=1,inplace=True)
f,ax=py.subplots(figsize=(23,20))
cormat=train.corr()
sns.heatmap(cormat,annot=True)
train.drop(["area_mean","radius_mean"],axis=1,inplace=True)
sns.swarmplot(x="diagnosis",y="smoothness_se",data=train)#smoothness_me is not useful for classifying the cancer 
train.drop(["smoothness_mean","smoothness_se","smoothness_worst"],axis=1,inplace=True)

sns.swarmplot(x="diagnosis",y="compactness_mean",data=train)
train.drop(["compactness_se","compactness_worst"],axis=1,inplace=True)
sns.swarmplot(x="diagnosis",y="concavity_mean",data=train)
train.drop(["compactness_mean","concave points_mean","concave points_worst"],axis=1,inplace=True)
sns.swarmplot(x="diagnosis",y="texture_mean",data=train)
train.drop(["concavity_se","concavity_worst"],axis=1,inplace=True)
train.drop(["area_se","radius_se"],axis=1,inplace=True)
train.drop("id",axis=1,inplace=True)
train["diagnosis"]=train["diagnosis"].map({"M":0,"B":1})
train.shape
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
train1,test=train_test_split(train,test_size=0.28)

train_x=train1.drop("diagnosis",axis=1)
train_y=train1["diagnosis"]
test_x=test.drop("diagnosis",axis=1)
y=test["diagnosis"]
lr=LogisticRegression()
lr.fit(train_x,train_y)
prd=lr.predict(test_x)
print("accuracy is ",metrics.accuracy_score(prd,y))
lr=SVC()
lr.fit(train_x,train_y)
prd=lr.predict(test_x)
print("accuracy is ",metrics.accuracy_score(prd,y))
lr=DecisionTreeClassifier()
lr.fit(train_x,train_y)
prd=lr.predict(test_x)
print("accuracy is ",metrics.accuracy_score(prd,y))
a=[]
ind=list(range(1,11))
for i in range(1,11):
    k=KNeighborsClassifier(n_neighbors=i)
    k.fit(train_x,train_y)
    prd=k.predict(test_x)
    a.append(metrics.accuracy_score(prd,y))
    
py.plot(ind,a)
k=KNeighborsClassifier(n_neighbors=4)
k.fit(train_x,train_y)
prd=k.predict(test_x)
print("accuracy is ",metrics.accuracy_score(prd,y))
    #a.append(metrics.accuracy_score(prd,y))
    
