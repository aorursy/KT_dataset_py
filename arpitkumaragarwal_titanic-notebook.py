# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.describe()
train.info()
#check target class
train['Survived'].value_counts()
train['Survived'].value_counts(normalize=True).plot.bar()
#EDA ON CATEGORICAL FEATURES
train['Pclass'].value_counts()
train['Pclass'].value_counts(normalize=True).plot.bar()
sns.countplot(x='Pclass',hue='Survived',data=train)
sns.countplot(x='Sex',hue='Survived',data=train)
sns.boxplot(y='Age',x='Sex',data=train)
train['Age'][train['Sex']=='male'].mean()
train['Age'][train['Sex']=='female'].mean()
train['SibSp'].value_counts()
sns.countplot(x='SibSp',hue='Survived',data=train)
train['Parch'].value_counts()
sns.countplot(x='Parch',hue='Survived',data=train)
train['Cabin'].value_counts().count()
sns.countplot(x='Cabin',hue='Survived',data=train)
sns.countplot(x='Embarked',hue='Survived',data=train)
sns.distplot(train['Fare'])
train.isnull().sum()
train_df=train.copy()
train['Sex'].replace('male',1,inplace=True)

train['Sex'].replace('female',0,inplace=True)

test['Sex'].replace('male',1,inplace=True)

test['Sex'].replace('female',0,inplace=True)
train.head()
def age(lt):

    a=lt[0]

    s=lt[1]

    if pd.isnull(a):

        if s==1:

            return 31

        elif s==0:

            return 28

    else:   

        return a

train['Age']=train[['Age','Sex']].apply(age,axis=1)
test['Age']=test[['Age','Sex']].apply(age,axis=1)
train['Age'].isnull().sum()
train['Cabin'].value_counts()
train.head()
train.isnull().sum()
train.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
train.head()
train.isnull().sum()
train['Embarked']=train['Embarked'].fillna(train['Embarked'].mode()[0])
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.title('Titanic Correlation of Features', y=1.05, size=15)

sns.heatmap(train.corr(), linewidths=0.1, vmax=1.0, 

            square=True, linecolor='white', annot=True)
train['Embarked'].replace('S',2,inplace=True)

train['Embarked'].replace('C',1,inplace=True)

train['Embarked'].replace('Q',0,inplace=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.title('Titanic Correlation of Features', y=1.05, size=15)

sns.heatmap(train.corr(), linewidths=0.1, vmax=1.0, 

            square=True, linecolor='white', annot=True)
train.head()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X=train.drop('Survived',axis=1).values

Y=train['Survived'].values

df_feat=scaler.fit_transform(X)
df_feat.shape
Y
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(df_feat,Y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression

logr=LogisticRegression()

logr.fit(xtrain,ytrain)

predict1_logr=logr.predict(xtest)

print(logr.score(xtest,ytest))
from sklearn.svm import SVC

svm=SVC()

svm.fit(xtrain,ytrain)

predict1_svm=svm.predict(xtest)

print(svm.score(xtest,ytest))
from sklearn.ensemble import RandomForestClassifier

tree=RandomForestClassifier()

tree.fit(xtrain,ytrain)

predict1_tree=tree.predict(xtest)

print(tree.score(xtest,ytest))
from sklearn.model_selection import cross_val_score
#Genaralised class
class Model:

    def __init__(self,model):

        self.model=model

        self.X,self.Y=X,Y

        self.xtrain,self.xtest,self.ytrain,self.ytest=xtrain,xtest,ytrain,ytest

        self.train()

        

    def model_name(self):

        model_name = type(self.model).__name__

        return model_name

    def train(self):

        print(f"Training {self.model_name()} Model...")

        self.model.fit(xtrain, ytrain)

        print("Model Trained.")

    

    def prediction(self, test_x=None, test=False):

        if test == False:

            y_pred = self.model.predict(self.xtest)

        else:

            y_pred = self.model.predict(test_x)

            

        return y_pred

    def accuracy(self):

        y_pred = self.prediction()

        y_test = self.ytest

        

        acc = accuracy_score(y_pred, ytest)

        print(f"{self.model_name()} Model Accuracy: ", acc)

        

    def cross_validation(self, cv=5):

        print(f"Evaluate {self.model_name()} score by cross-validation...")

        CVS = cross_val_score(self.model, self.X, self.Y, scoring='accuracy', cv=cv)

        print(CVS)

        print("="*60, "\nMean accuracy of cross-validation: ", CVS.mean())    

    

    

    

    

    
logr2=LogisticRegression()

logr2=Model(logr2)
logr2.accuracy()
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(ytest,  predict1_logr) 

auc = metrics.roc_auc_score(ytest, predict1_logr)

plt.figure(figsize=(12,8)) 

plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 

plt.xlabel('False Positive Rate') 

plt.ylabel('True Positive Rate')

plt.legend(loc=4)

plt.show()
#Boosting
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

dtt=DecisionTreeClassifier()

abcf=AdaBoostClassifier(n_estimators=100,base_estimator=dt,learning_rate=1,random_state=2)

adcf=Model(abcf)
adcf.accuracy()
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb=Model(xgb)

xgb.accuracy()
xgb.cross_validation()


CVS = cross_val_score(xgb, X, Y, scoring='accuracy', cv=5)

print(CVS)

print("="*60, "\nMean accuracy of cross-validation: ", CVS.mean())
class Model1:

    def __init__(self,model):

        self.model=model

        self.X,self.Y=X,Y

        self.xtrain,self.xtest,self.ytrain,self.ytest=xtrain,xtest,ytrain,ytest

        self.train()

        

    def model_name(self):

        model_name = type(self.model).__name__

        return model_name

    def train(self):

        print(f"Training {self.model_name()} Model...")

        self.model.fit(xtrain, ytrain)

        print("Model Trained.")

    

    def prediction(self, test_x=None, test=False):

        if test == False:

            y_pred = self.model.predict(self.xtest)

        else:

            y_pred = self.model.predict(test_x)

            

        return y_pred

    def accuracy(self):

        y_pred = self.prediction()

        y_test = self.ytest

        

        acc = accuracy_score(y_pred, ytest)

        print(f"{self.model_name()} Model Accuracy: ", acc)

        

    def cross_validation(self, cv=5):

        print(f"Evaluate {self.model_name()} score by cross-validation...")

        CVS = cross_val_score(self.model, self.X, self.Y, scoring='accuracy', cv=cv)

        print(CVS)

        print("="*60, "\nMean accuracy of cross-validation: ", CVS.mean())    

    

    

    

    

    
xgb1=XGBClassifier()

xgb1=Model1(xgb1)
xgb1.accuracy()
xgb1.cross_validation()
adda=AdaBoostClassifier(n_estimators=100,base_estimator=dt,learning_rate=1)

adda=Model1(adda)

adda.accuracy()
adda.cross_validation()
logis=LogisticRegression()

logis=Model1(logis)

logis.accuracy()
logis.cross_validation()
test.head()
test.isnull().sum()
test1=test.copy()

test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
test.head()
test['Fare']=test['Fare'].fillna(test['Fare'].mean())
test['Embarked'].replace('S',2,inplace=True)

test['Embarked'].replace('C',1,inplace=True)

test['Embarked'].replace('Q',0,inplace=True)
Xtest=scaler.fit_transform(test.values)
Xtest
xgb_pred = xgb1.prediction(test_x=Xtest, test=True)
submit=pd.DataFrame()

submit['PassengerId']=test1['PassengerId']

submit['Survived']=xgb_pred
submit.to_csv('submission_my.csv',index=False)
xgb2=XGBClassifier()

xgb2.fit(X,Y)

predict_xgb=xgb2.predict(Xtest)
submit2=pd.DataFrame()

submit2['PassengerId']=test1['PassengerId']

submit2['Survived']=predict_xgb
submit2.to_csv('submission_my2.csv',index=False)