
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
dft=pd.read_csv('/kaggle/input/titanic/test.csv')
dft.head()
print(df.shape)
print(dft.shape)

df.describe()
dft.describe()
df.info()
dft.info()
df.isnull().sum()
dft.isnull().sum()
#visualise missing value 
import missingno as mn
mn.matrix(df)
mn.matrix(dft)
df = df.drop(['Cabin'], axis = 1)

dft = dft.drop(['Cabin'], axis = 1)
df['Embarked'].value_counts()
#Embarked has 2 missing value fill with S which has highest number
df['Embarked'].fillna('S',inplace=True)
dft['Embarked'].fillna('S',inplace=True)
#replace NaN value in Age with mean value
median=np.round(df['Age'].median(),1)
df['Age'].fillna(median,inplace=True)
median=np.round(dft['Age'].median(),1)
dft['Age'].fillna(median,inplace=True)
df.isnull().sum()
dft.isnull().sum()
#replace Sex column with numeric 0,1 with male,female rep
df=df.replace({'male': 0,
            'female' : 1})
df.head()
dft=dft.replace({'male': 0,
            'female' : 1})
dft.head()
#find the correlation between data
df.corr()
#visualise correlation data using heatmap
plt.figure(figsize=(14,6))
sns.heatmap(df.corr(),annot=True)

#plot graph between pclass and survived
sns.barplot(x='Pclass',y='Survived',data=df)
#Sex vs Survived
sns.barplot(x='Sex',y='Survived',data=df)
#Embarked vs Survived
sns.barplot(x='Embarked',y='Survived',data=df)
#Drop unwanted columns such as Name,Ticket,Fare is decided by Pclass so drop fare also
df=df.drop(['Name','Ticket','Fare'],axis=1)
dft=dft.drop(['Name','Ticket','Fare'],axis=1)
#add SibSp and Parch in Family
df['Family']=df['SibSp']+df['Parch']+1
df=df.drop(['SibSp','Parch'],axis=1)
dft['Family']=dft['SibSp']+dft['Parch']+1
dft=dft.drop(['SibSp','Parch'],axis=1)
#Categorise Age
def AgeGroup(age):
    a=''
    if age<=10:
        a='Child'
    elif age<=30:
        a='Young'
    elif age<=50:
        a='Adult'
    else:
        a='Old'
    return a
df['AgeGroup']=df['Age'].map(AgeGroup)
df=df.drop(['Age'],axis=1)
dft['AgeGroup']=dft['Age'].map(AgeGroup)
dft=dft.drop(['Age'],axis=1)
#Categorise Family
def FamilyGroup(family):
    a=''
    if family<=1:
        a='Solo'
    elif family<=4:
        a='Small'
    else:
        a='Large'
    return a
df['FamilyGroup']=df['Family'].map(FamilyGroup)
df=df.drop(['Family'],axis=1)    
dft['FamilyGroup']=dft['Family'].map(FamilyGroup)
dft=dft.drop(['Family'],axis=1)
#get dummies variable
df=pd.get_dummies(df,columns=['Embarked','AgeGroup','FamilyGroup','Sex'])
dft=pd.get_dummies(dft,columns=['Embarked','AgeGroup','FamilyGroup','Sex'])
print(df.shape)
print(dft.shape)
df.head()
dft.head()
X=df.drop(['Survived'],axis=1)
X.head()
y=df['Survived']
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)
#import all lib
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
#KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X_train,y_train)
y_pred=LR.predict(X_test)
print("The best accuracy with LR is", metrics.accuracy_score(y_test,y_pred))

#SVM
from sklearn import svm
SVM=svm.SVC().fit(X_train,y_train)
y_pred=SVM.predict(X_test)
print("The best accuracy with SVM is", metrics.accuracy_score(y_test,y_pred))
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
def getaccuracy(max_leaf,X_train,y_train,X_test,y_test):
    DT=DecisionTreeClassifier().fit(X_train,y_train)
    y_pred=DT.predict(X_test)
    return(metrics.accuracy_score(y_test,y_pred))
    

for max_leaf in [5,50,500]:
    my_mae = getaccuracy(max_leaf,X_train,y_train,X_test,y_test)
    print("Max leaf : ",max_leaf,'The best accuracy with SVM is',my_mae)
#define whole train as TrainX and Trainy
TrainX=df.drop(['Survived'],axis=1)
Trainy=df['Survived']
from sklearn import svm
SVM=svm.SVC().fit(TrainX,Trainy)
y_pred=SVM.predict(dft)
submission = pd.DataFrame({
        "PassengerId": dft["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titanic.csv', index=False)
print("Submitted Successfully")
