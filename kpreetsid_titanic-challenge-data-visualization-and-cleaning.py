import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
%matplotlib inline
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
#Calculating the number of male/female passenger Survived.
sns.countplot(x='Survived',data=train_df,hue='Sex')
#Plotting the percentage of passengers survived according to the Class they were in. 
sns.factorplot(x='Pclass',data=train_df,kind='count',hue='Survived')
#Further breaking the above graph to male/female level
sns.factorplot(x='Survived',data=train_df,hue='Sex',kind='count',col='Pclass')
#Age distribution of the passengers
sns.distplot(train_df['Age'].dropna(),bins=30,kde=False)
#Survivers according to their gender and Pclass
sns.factorplot(x='Pclass',y='Survived',data=train_df,hue='Sex')
train_df.info()
print('_'*40)
test_df.info()
#Dropping Cabin column from both datasets
train_df.drop(['Cabin'],inplace=True,axis=1)
test_df.drop(['Cabin'],inplace=True,axis=1)
train_df['Embarked']=train_df['Embarked'].fillna('S')
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].mean())
train_df.head()
plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=train_df)
def age_mean(x):
    Age,Pclass=x
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 28
        else:
            return 24
    else:
        return Age
train_df['Age']=train_df[['Age','Pclass']].apply(age_mean,axis=1)
plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=test_df)
def age_mean_test(x):
    Age,Pclass=x
    if pd.isnull(Age):
        if Pclass==1:
            return 43
        elif Pclass==2:
            return 26
        else:
            return 25
    else:
        return Age
test_df['Age']=test_df[['Age','Pclass']].apply(age_mean_test,axis=1)
plt.figure(figsize=(10,6))
sns.heatmap(train_df.isnull())
plt.figure(figsize=(10,6))
sns.heatmap(test_df.isnull())
def m_f(x):
    Sex=x
    if Sex=='male':
        return 1
    else:
        return 0
train_df['Sex']=train_df['Sex'].apply(m_f)
test_df['Sex']=test_df['Sex'].apply(m_f)
train_df.head()
def name(x):
    Name=x
    if Name=='Mr.':
        return 'Mr'
    elif Name=='Miss.':
        return 'Miss'
    elif Name=='Mrs.':
        return 'Mrs'
    else:
        return 'other'
train_df['Name']=train_df['Name'].map(lambda x: x.split(' ')[1])
train_df['Name']=train_df['Name'].apply(name)
test_df['Name']=test_df['Name'].map(lambda x: x.split(' ')[1])
test_df['Name']=test_df['Name'].apply(name)
train_df.info()
print('_'*40)
test_df.info()
train_df.head()
test_df.head()
nametrain=pd.get_dummies(train_df['Name'],drop_first=True)
nametest=pd.get_dummies(test_df['Name'],drop_first=True)
embarkedtrain=pd.get_dummies(train_df['Embarked'],drop_first=True)
embarkedtest=pd.get_dummies(test_df['Embarked'],drop_first=True)
pclasstrain=pd.get_dummies(train_df['Pclass'],drop_first=True)
pclasstest=pd.get_dummies(test_df['Pclass'],drop_first=True)
tr_df=pd.concat([train_df,nametrain,embarkedtrain,pclasstrain],axis=1)
te_df=pd.concat([test_df,nametest,embarkedtest,pclasstest],axis=1)
te_df.drop(['Name','Embarked','Pclass','Ticket'],axis=1,inplace=True)
tr_df.drop(['Name','Embarked','Pclass','Ticket'],axis=1,inplace=True)
tr_df.head()
#Applying Mean Normalization to both datasets
tr_df['Age']=(tr_df['Age']-tr_df['Age'].mean())/(tr_df['Age'].max()-tr_df['Age'].min())
tr_df['Fare']=(tr_df['Fare']-tr_df['Fare'].mean())/(tr_df['Fare'].max()-tr_df['Fare'].min())
tr_df.head()
te_df['Age']=(te_df['Age']-te_df['Age'].mean())/(te_df['Age'].max()-te_df['Age'].min())
te_df['Fare']=(te_df['Fare']-te_df['Fare'].mean())/(te_df['Fare'].max()-te_df['Fare'].min())
tr_df.head()
x_train=tr_df[['Sex', 'Age', 'SibSp', 'Parch','Fare', 'Mr', 'Mrs', 'other', 'Q', 'S',2,3]]
y_train=tr_df['Survived']
x_test=te_df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Mr','Mrs', 'other', 'Q', 'S',2,3]]
#Data looks clean and nice, it's time for the model training.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.score(x_train,y_train)
svc=SVC()
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
rnf=RandomForestClassifier()
rnf.fit(x_train,y_train)
rnf.score(x_train,y_train)










a