import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/train.csv')

data.head()
data.info()
data.describe()
data.isnull().sum()
plt.figure(figsize = (8,6))
sns.countplot(x='Embarked',data=data)
data['Embarked'][data['Embarked'].isnull()]='S'
data['Embarked'].isnull().sum()
sns.boxplot(x='Pclass',y='Age',data=data)
data['Age'][data['Age'].isnull()]=data['Age'].mean()
data['Age'].isnull().sum()
data['Has Cabin']= data['Cabin'].apply(lambda x:0 if isinstance(x,float) else 1)
data['Survived'].isnull().sum()
data['Survived'].fillna(value=0,inplace=True)
data['Fare'][data['Fare'].isnull()]= data['Fare'].mean()
data.isnull().sum()
sns.countplot(x='Survived',data=data,hue='Sex')
print('Percentage of Male Survived: ',data['Survived'][data['Sex']=='male'].value_counts(normalize=True)[1]*100)
print('Percentage of Female Survived: ',data['Survived'][data['Sex']=='female'].value_counts(normalize=True)[1]*100)
sns.barplot(x='Pclass',y='Survived',data=data)
print('Percentage of class 1 passenger Survived: ',data['Survived'][data['Pclass']== 1].value_counts(normalize=True)[1]*100)
print('Percentage of class 2 passenger Survived: ',data['Survived'][data['Pclass']== 2].value_counts(normalize=True)[1]*100)
print('Percentage of class 3 passenger Survived: ',data['Survived'][data['Pclass']== 3].value_counts(normalize=True)[1]*100)

sns.barplot(x='Has Cabin',y='Survived',data=data)
print('Percentage of passengers have Cabin & Survived: ',data['Survived'][data['Has Cabin']== 1].value_counts(normalize=True)[1]*100)
print('Percentage of passengers dont have Cabin & Survived: ',data['Survived'][data['Has Cabin']== 0].value_counts(normalize=True)[1]*100)
sns.countplot(x='Embarked',data=data)
print('percentage of P embarked from S: ', ((data['Embarked']=='S').value_counts())/(data['Embarked'].count())*100)
print('percentage of P embarked from C: ', ((data['Embarked']=='C').value_counts())/(data['Embarked'].count())*100)
print('percentage of P embarked from Q: ', ((data['Embarked']=='Q').value_counts())/(data['Embarked'].count())*100)

sns.barplot(x='Embarked',y='Survived',data=data)
print('Percentage of S embarked survived: ',data['Survived'][data['Embarked']=='S'].value_counts(normalize=True)[1]*100)
print('Percentage of Q embarked survived: ',data['Survived'][data['Embarked']=='Q'].value_counts(normalize=True)[1]*100)
print('Percentage of C embarked survived: ',data['Survived'][data['Embarked']=='C'].value_counts(normalize=True)[1]*100)

import random
random.sample(list(data['Name'].values),10)
data['Title']=Titles=data['Name'].apply(lambda x: x.split(',')[1].split('.')[0] if ',' in x else x)
data['Title'].value_counts()
def map_marriage(Title):
    Title = Title.strip()
    if Title in ['Dr', 'Col', 'Capt','Major','Don','Rev','Dona','Jonkheer']:
        return 0
    if Title in ['the Countess', 'Lady', 'Sir']:
        return 1
    if Title in ['Mlle','Ms','Miss']:
        return 2
    if Title in ['Mrs']:
        return 3
    if Title in ['Mr','Master','Mme']:
        return 4
data['Title']=data['Title'].apply(map_marriage)
data.head()
data['Male']=data['Sex'].map({'male':1,'female':0})
data['FamSize']=data['SibSp'] + data['Parch']+1
def map_age(Age):
    if Age <=12:
        return 'Child'
    elif 12 < Age <=18:
        return 'Teenager'
    elif 18 < Age <=50:
        return 'Adult'
    else:
        return 'Old'
data['Age']=data['Age'].apply(map_age)
data['Age'] = data['Age'].map({'Child':1,'Teenager':2,'Adult':3,'Old':4})
def impute_fare(Fare):
    if Fare <=20:
        return 1
    if 20 < Fare <=40:
        return 2
    if Fare > 40:
        return 3
data['Fare'] = data['Fare'].apply(impute_fare)
data['Embarked'] = data['Embarked'].map({'S':1,'C':2,'Q':3})
data.head()
df_data = data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Has Cabin','SibSp','Parch'],axis=1)
df_data.head()
from sklearn.model_selection import train_test_split
X=df_data.drop(['Survived'],axis=1)
y= df_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,random_state=50)
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)
pred_logmodel = logmodel.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
acc_logmodel = round(accuracy_score(pred_logmodel,y_test)*100,2)
print('Accuracy of Logmodel is: ',acc_logmodel)
print(classification_report(pred_logmodel,y_test))
from sklearn.svm import SVC
SVM_model = SVC()
SVM_model.fit(X_train,y_train)
pred_SVM = SVM_model.predict(X_test)
acc_SVM_model = round(accuracy_score(pred_SVM,y_test)*100,2)
print('Accuracy of SVM_model is: ',acc_SVM_model)
print(classification_report(pred_logmodel,y_test))
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train,y_train)
pred_DT = DT_model.predict(X_test)
acc_DT_model = round(accuracy_score(pred_DT,y_test)*100,2)
print('Accuracy of DT_model is: ',acc_DT_model)
print(classification_report(pred_DT,y_test))
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(X_train,y_train)
pred_RF = RF_model.predict(X_test)
acc_RF_model = round(accuracy_score(pred_RF,y_test)*100,2)
print('Accuracy of RF_model is: ',acc_RF_model)
print(classification_report(pred_RF,y_test))
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train,y_train)
pred_KNN = KNN_model.predict(X_test)
acc_KNN_model = round(accuracy_score(pred_KNN,y_test)*100,2)
print('Accuracy of KNN_model is: ',acc_KNN_model)
print(classification_report(pred_KNN,y_test))
models = pd.DataFrame({'Model':['Support Vector Machines','Random Forest','Decision Tree','Logistic Regression','K Nearest'],
                      'Score':[acc_SVM_model,acc_RF_model,acc_DT_model,acc_logmodel,acc_KNN_model]
                      }).sort_values(by='Score',ascending=False)
models
test_data = pd.read_csv('../input/test.csv')
test_data.head()
test_data.info()
test_data['Age'][test_data['Age'].isnull()] = test_data['Age'].mean()
test_data['Fare'][test_data['Fare'].isnull()] = test_data['Fare'].mean()
test_data.info()
import random
random.sample(list(test_data['Name'].values),10)
test_data['Title']=Titles=test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0] if ',' in x else x)
test_data['Title'].value_counts()
def map_marriage(Title):
    Title = Title.strip()
    if Title in ['Dr', 'Col', 'Capt','Major','Don','Rev','Dona','Jonkheer']:
        return 0
    if Title in ['the Countess', 'Lady', 'Sir']:
        return 1
    if Title in ['Mlle','Ms','Miss']:
        return 2
    if Title in ['Mrs']:
        return 3
    if Title in ['Mr','Master','Mme']:
        return 4
test_data['Title'] = test_data['Title'].apply(map_marriage)
test_data['Male']=test_data['Sex'].map({'male':1,'female':0})
def map_age(Age):
    if Age <=12:
        return 'Child'
    if 12 < Age <=18:
        return 'Teenager'
    if 18 < Age <=50:
        return 'Adult'
    if Age >50:
        return 'Old'
test_data['Age'] = test_data['Age'].apply(map_age)
test_data['Age'] = test_data['Age'].map({'Child':1,'Teenager':2,'Adult':3,'Old':4})
test_data.head()
test_data['Embarked'] = test_data['Embarked'].map({'S':1,'Q':2,'C':3})
def impute_fare(Fare):
    if Fare <=20:
        return 1
    if 20 < Fare <=40:
        return 2
    if Fare > 40:
        return 3
test_data['Fare'] = test_data['Fare'].apply(impute_fare)
test_data['Fare'].unique()
test_data['FamSize'] = test_data['SibSp'] + test_data['Parch'] +1
test_data['Has Cabin'] = test_data['Cabin'].apply(lambda x: 0 if isinstance(x,float) else 1)
test_data['Has Cabin'][test_data['Has Cabin']==1].count()
df_test = test_data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Has Cabin','SibSp','Parch'],axis=1)
test_data['Survived'] = RF_model.predict(df_test)
test_data.head(15)
submission = test_data[['PassengerId','Survived']]
submission.head(15)
submission.to_csv('New Submission.csv',index=False)
