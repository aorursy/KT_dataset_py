import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
import os
print("")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train.info()
test.info()
train.shape[0]       
(train.isnull().sum()/train.shape[0])*100
(test.isnull().sum()/test.shape[0])*100
train.describe()
test.describe()
train_drop=train.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)
test_drop=test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)
train.head()
test.head()
train['Survived'].value_counts()
sns.pairplot(train,x_vars=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],y_vars='Survived',size=2,aspect=1)
plt.show()
train.isnull().sum()
train['Age'].fillna(train['Age'].median(),inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
train['Fare'].fillna(train['Fare'].median(),inplace=True)
train.isnull().sum()
train.dropna(inplace=True)
test.isnull().sum()
test['Age'].fillna(test['Age'].median(),inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)
test['Fare'].fillna(test['Fare'].median(),inplace=True)
# test.dropna(inplace=True)
test.isnull().sum()
test.shape
bxplot=train.select_dtypes(include=['float64','int64'])
bxplot.drop('Survived',axis=1,inplace=True)
bxcols=bxplot.columns
bxcols
def bxplott(df): 
    for i in bxcols:
            sns.boxplot(data=df,x=df[i])
            plt.show()
bxplott(train)
Q3=train['Pclass'].quantile(0.85)
Q1=train['Pclass'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Pclass'] >= Q1 - 1.5*IQR) & (train['Pclass']<= Q3 + 1.5*IQR) ]

Q3=train['Age'].quantile(0.85)
Q1=train['Age'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Age'] >= Q1 - 1.5*IQR) & (train['Age']<= Q3 + 1.5*IQR) ]

Q3=train['SibSp'].quantile(0.85)
Q1=train['SibSp'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['SibSp'] >= Q1 - 1.5*IQR) & (train['SibSp']<= Q3 + 1.5*IQR) ]

Q3=train['Parch'].quantile(0.85)
Q1=train['Parch'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Parch'] >= Q1 - 1.5*IQR) & (train['Parch']<= Q3 + 1.5*IQR) ]

Q3=train['Fare'].quantile(0.85)
Q1=train['Fare'].quantile(0.15)
IQR=Q3-Q1
train=train[(train['Fare'] >= Q1 - 1.5*IQR) & (train['Fare']<= Q3 + 1.5*IQR) ]
train.head()
def bxplott(df): 
      for i in bxcols:
            sns.boxplot(data=df,x=df[i])
            plt.show()
bxplott(train)
train['Sex']=train['Sex'].map({'male':0,'female':1})
test['Sex']=test['Sex'].map({'male':0,'female':1})
embarked_train=pd.get_dummies(train['Embarked'],prefix='Embarked',drop_first=True)
train.drop(["Embarked"],axis=1,inplace=True)
train=pd.concat([train,embarked_train],axis=1)
embarked_test=pd.get_dummies(test['Embarked'],prefix='Embarked',drop_first=True)
test.drop(["Embarked"],axis=1,inplace=True)
test=pd.concat([test,embarked_test],axis=1)
train.head()
test.head()
train['SibSp'].value_counts()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scalecols=['Pclass','Age','Fare','SibSp','Parch']
train[scalecols]=scaler.fit_transform(train[scalecols])
train.head()
scalecols_test=['Pclass','Age','Fare','SibSp','Parch']
test[scalecols_test]=scaler.transform(test[scalecols_test])
test.head()
train_PassengerId = train['PassengerId']
test_PassengerId = test['PassengerId']
test.drop(['PassengerId'],inplace = True,axis=1)
train.drop(['PassengerId'],inplace = True,axis=1)
y=train['Survived']
X=train.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)
X_train.head()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print("Cross_Val_Score:",cross_val_score(logreg,X_train,y_train,cv = 5,scoring = 'accuracy').mean())
converted=(sum(train['Survived'])/len(train['Survived'].index)*100)
print(converted)
# Importing Libraries
from sklearn.ensemble import RandomForestClassifier

# Creating an object of RandomForestClassifier class with name rf_model and fitting the model on X_train and y_train
rf_model = RandomForestClassifier(n_estimators=1000)
rf_model.fit(X_train, y_train)
# Predicting on y value 
y_pred = rf_model.predict(test)
# Creating a new dataframe which can store value of listing id and exporting it in csv format
submission = pd.DataFrame()
submission["PassengerId"] = test_PassengerId
submission["Survived"] = y_pred
submission.to_csv("Submission.csv", index=False)