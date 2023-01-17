#data analysis libraries 
import numpy as np
import pandas as pd
import random

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
data_train=pd.read_csv("../input/train.csv") # train data
data_test=pd.read_csv("../input/test.csv") # test data

print("Train info:\n")
data_train.info()
print("-"*40)
print("Test info:\n")
data_test.info()
data_train.head()
print('Train columns with null values:\n',data_train.isnull().sum()) # sum. of null values
print("-"*40)
print('Test columns with null values:\n',data_test.isnull().sum())
data_train.describe(include = 'all')
print(data_train['Survived'].value_counts())

sns.set()
f,ax=plt.subplots(1,2,figsize=(12,5))
data_train['Survived'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Survived',color = 'r',fontsize=15)
ax[0].set_ylabel('')

sns.countplot('Survived',data=data_train,ax=ax[1])
ax[1].set_title('Rate of the Survived',color = 'r',fontsize=15)

plt.show()
print(data_train.groupby(['Pclass','Survived'])['Survived'].count())

f,ax=plt.subplots(1,3,figsize=(20,5))

data_train.groupby(['Pclass','Survived'])['Survived'].count().plot.pie(autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Survived vs Dead by Pclass',color = 'r',fontsize=15)
ax[0].set_ylabel('')

sns.countplot('Survived',data=data_train,hue='Pclass',ax=ax[1])
ax[1].set_title('Survived vs Dead by Pclass',color = 'r',fontsize=15)

sns.barplot(x=data_train.groupby(['Pclass'])['Survived'].mean().index,y=data_train.groupby(['Pclass'])['Survived'].mean().values,ax=ax[2])
ax[2].set_title('Rate of the Survived by Pclass',color = 'r',fontsize=15)

plt.show()
print(data_train.groupby(['Sex','Survived'])['Survived'].count())

f,ax=plt.subplots(1,3,figsize=(20,5))

data_train.groupby(['Sex','Survived'])['Survived'].count().plot.pie(autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Survived vs Dead by Sex',color = 'r',fontsize=15)
ax[0].set_ylabel('')

sns.countplot('Survived',data=data_train,hue='Sex',ax=ax[1])
ax[1].set_title('Survived vs Dead by Sex',color = 'r',fontsize=15)

sns.barplot(x=data_train.groupby(['Sex'])['Survived'].mean().index,y=data_train.groupby(['Sex'])['Survived'].mean().values,ax=ax[2])
ax[2].set_title('Rate of the Survived by Sex',color = 'r',fontsize=15)

plt.show()
print('Age of the oldest passanger:',data_train['Age'].max(),'years old')
print('Age of the youngest passanger:',data_train['Age'].min(),'years old')
print('Average Age on the ship:',data_train['Age'].mean(),'years old')

f,ax=plt.subplots(1,3,figsize=(20,5))

data_train[data_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='b')
ax[0].set_title('Dead',color = 'r',fontsize=15)

data_train[data_train['Survived']==1].Age.plot.hist(ax=ax[1],color='orange',bins=20,edgecolor='black')
ax[1].set_title('Survived',color = 'r',fontsize=15)

sns.kdeplot(data_train["Age"][(data_train["Survived"] == 0) & (data_train["Age"].notnull())], color="Red", shade = True,ax=ax[2])
sns.kdeplot(data_train["Age"][(data_train["Survived"] == 1) & (data_train["Age"].notnull())], color="Blue", shade= True,ax=ax[2])
ax[2].set_title('Rate of Survived vs Dead',color = 'r',fontsize=15)
ax[2].legend(['Dead','Survived'])

plt.show()
print(data_train.groupby(['SibSp','Survived'])['Survived'].count())

f,ax=plt.subplots(1,3,figsize=(20,5))

data_train[data_train['Survived']==0].SibSp.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='b')
ax[0].set_title('Dead',color = 'r',fontsize=15)

data_train[data_train['Survived']==1].SibSp.plot.hist(ax=ax[1],color='orange',bins=20,edgecolor='black')
ax[1].set_title('Survived',color = 'r',fontsize=15)

sns.factorplot('SibSp','Survived',data=data_train,ax=ax[2])
ax[2].set_title('Rate of the Survived',color = 'r',fontsize=15)
plt.close(2)

plt.show()
print(data_train.groupby(['Parch','Survived'])['Survived'].count())

f,ax=plt.subplots(1,3,figsize=(20,5))

data_train[data_train['Survived']==0].Parch.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='b')
ax[0].set_title('Dead',color = 'r',fontsize=15)

data_train[data_train['Survived']==1].Parch.plot.hist(ax=ax[1],color='orange',bins=20,edgecolor='black')
ax[1].set_title('Survived',color = 'r',fontsize=15)

sns.factorplot('Parch','Survived',data=data_train,ax=ax[2])
ax[2].set_title('Rate of the Survived',color = 'r',fontsize=15)
plt.close(2)

plt.show()
data_train['Ticket'].describe()
print('The highest fare was:',data_train['Fare'].max())
print('The lowest fare was:',data_train['Fare'].min())
print('The avarage was:',data_train['Fare'].mean())

f,ax=plt.subplots(1,3,figsize=(20,5))

data_train[data_train['Survived']==0].Fare.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='b')
ax[0].set_title('Dead',color = 'r',fontsize=15)

data_train[data_train['Survived']==1].Fare.plot.hist(ax=ax[1],color='orange',bins=20,edgecolor='black')
ax[1].set_title('Survived',color = 'r',fontsize=15)

sns.kdeplot(data_train["Fare"][(data_train["Survived"] == 0) & (data_train["Age"].notnull())], color="Red", shade = True,ax=ax[2])
sns.kdeplot(data_train["Fare"][(data_train["Survived"] == 1) & (data_train["Age"].notnull())], color="Blue", shade= True,ax=ax[2])
ax[2].set_title('Rate of the Survived',color = 'r',fontsize=15)
ax[2].legend(['Dead','Survived'])

plt.show()
data_train['Cabin'].describe()
print(data_train.groupby(['Embarked','Survived'])['Survived'].count())

f,ax=plt.subplots(1,3,figsize=(20,5))

data_train.groupby(['Embarked','Survived'])['Survived'].count().plot.pie(autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Survived vs Dead by Embarked',color = 'r',fontsize=15)
ax[0].set_ylabel('')

sns.countplot('Survived',data=data_train,hue='Embarked',ax=ax[1])
ax[1].set_title('Survived vs Dead by Embarked',color = 'r',fontsize=15)

sns.barplot(x=data_train.groupby(['Embarked'])['Survived'].mean().index,y=data_train.groupby(['Embarked'])['Survived'].mean().values,ax=ax[2])
ax[2].set_title('Rate of the Survived by Embarked',color = 'r',fontsize=15)

plt.show()
Title_train=[i.split(",")[1].split(".")[0].strip() for i in data_train["Name"]] # split names from , to .
data_train["Title"] = pd.Series(Title_train)

Title_test=[i.split(",")[1].split(".")[0].strip() for i in data_test["Name"]]
data_test["Title"] = pd.Series(Title_test)

Title=pd.concat([data_train[['Title','Sex']],data_test[['Title','Sex']]],axis=0) 

print(Title.groupby(['Title','Sex'])['Title'].count())

plt.figure(figsize=(15,5))

sns.barplot(x=Title["Title"].value_counts().index,y=Title["Title"].value_counts().values)
plt.xticks(rotation=45)
plt.title('Titles',color = 'r',fontsize=15)

plt.show()
data_train['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                            'Rare',inplace=True)
data_train['Title'].replace(['Mlle','Mme','Ms'], 'Miss',inplace=True)

data_test['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                           'Rare',inplace=True)
data_test['Title'].replace(['Mlle','Mme','Ms'], 'Miss',inplace=True)

data_train.head()
Age=pd.concat([data_train[['Title','Age']],data_test[['Title','Age']]],axis=0)

print(Age.groupby('Title')['Age'].mean())

data_train.loc[(data_train['Age'].isnull())&(data_train['Title']=='Master'),'Age'] = Age[Age['Title']=='Master'].Age.mean()
data_train.loc[(data_train['Age'].isnull())&(data_train['Title']=='Miss'),'Age'] = Age[Age['Title']=='Miss'].Age.mean()
data_train.loc[(data_train['Age'].isnull())&(data_train['Title']=='Mr'),'Age'] = Age[Age['Title']=='Mr'].Age.mean()
data_train.loc[(data_train['Age'].isnull())&(data_train['Title']=='Mrs'),'Age'] = Age[Age['Title']=='Mrs'].Age.mean()
data_train.loc[(data_train['Age'].isnull())&(data_train['Title']=='Rare'),'Age'] = Age[Age['Title']=='Rare'].Age.mean()

data_test.loc[(data_test['Age'].isnull())&(data_test['Title']=='Master'),'Age'] = Age[Age['Title']=='Master'].Age.mean()
data_test.loc[(data_test['Age'].isnull())&(data_test['Title']=='Miss'),'Age'] = Age[Age['Title']=='Miss'].Age.mean()
data_test.loc[(data_test['Age'].isnull())&(data_test['Title']=='Mr'),'Age'] = Age[Age['Title']=='Mr'].Age.mean()
data_test.loc[(data_test['Age'].isnull())&(data_test['Title']=='Mrs'),'Age'] = Age[Age['Title']=='Mrs'].Age.mean()
data_test.loc[(data_test['Age'].isnull())&(data_test['Title']=='Rare'),'Age'] = Age[Age['Title']=='Rare'].Age.mean()
f,ax=plt.subplots(1,3,figsize=(23,6))

data_train.groupby(['Title','Survived'])['Survived'].count().plot.pie(autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Survived vs Dead by Title',color = 'r',fontsize=15)
ax[0].set_ylabel('')

sns.countplot('Survived',data=data_train,hue='Title',ax=ax[1])
ax[1].set_title('Survived vs Dead by Title',color = 'r',fontsize=15)

sns.barplot(x=data_train.groupby(['Title'])['Survived'].mean().index,y=data_train.groupby(['Title'])['Survived'].mean().values,ax=ax[2])
ax[2].set_title('Rate of the Survived by Title',color = 'r',fontsize=15)

plt.show()
data_test[data_test['Fare'].isnull()]
Fare=pd.concat([data_train[['Fare','Pclass','Embarked','Parch','Sex','SibSp','Title']],
                data_test[['Fare','Pclass','Embarked','Parch','Sex','SibSp','Title']]],axis=0)

data_test['Fare'].fillna(Fare[(Fare["Pclass"]==3) & (Fare["Embarked"]=='S') & (Fare["SibSp"]==0) & 
           (Fare["Parch"]==0) & (Fare["Sex"]=='male') & (Fare["Title"]=='Mr')].Fare.median(),inplace=True)

data_test.iloc[152]
data_train['Cabin'] = data_train['Cabin'].str[0] # add initial value to same location

data_test['Cabin'] = data_test['Cabin'].str[0]

Cabin=pd.concat([data_train[['Cabin','Embarked','Pclass','Fare']],data_test[['Cabin','Embarked','Pclass','Fare']]],axis=0)

Cabin.groupby(['Pclass','Embarked','Cabin'])['Fare'].max()
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='C')&(data_train.Fare<=56.9292),'Cabin']='A'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='C')&(data_train.Fare>56.9292)&(data_train.Fare<=113.2750),'Cabin']='D'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='C')&(data_train.Fare>113.2750)&(data_train.Fare<=134.5000),'Cabin']='E'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='C')&(data_train.Fare>134.5000)&(data_train.Fare<=227.5250),'Cabin']='C'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='C')&(data_train.Fare>227.5250),'Cabin']='B'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='S')&(data_train.Fare<=35.5000),'Cabin']='T'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='S')&(data_train.Fare>35.5000)&(data_train.Fare<=77.9583),'Cabin']='D'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='S')&(data_train.Fare>77.9583)&(data_train.Fare<=79.6500),'Cabin']='E'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='S')&(data_train.Fare>79.6500)&(data_train.Fare<=81.8583),'Cabin']='A'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='S')&(data_train.Fare>81.8583)&(data_train.Fare<=211.3375),'Cabin']='B'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&
               (data_train.Embarked=='S')&(data_train.Fare>211.3375),'Cabin']='C'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==1)&(data_train.Embarked=='Q'),'Cabin']='C'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==2)&
               (data_train.Embarked=='S')&(data_train.Fare<=13.0000),'Cabin']=random.sample(['D','E'],1)
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==2)&
               (data_train.Embarked=='S')&(data_train.Fare>13.0000),'Cabin']='F'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==2)&(data_train.Embarked=='C'),'Cabin']='D'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==2)&(data_train.Embarked=='Q'),'Cabin']='E'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==3)&
               (data_train.Embarked=='S')&(data_train.Fare<=7.6500),'Cabin']='F'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==3)&
               (data_train.Embarked=='S')&(data_train.Fare>7.6500)&(data_train.Fare<=12.4750),'Cabin']='E'
data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==3)&
               (data_train.Embarked=='S')&(data_train.Fare>12.4750),'Cabin']='G'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==3)&(data_train.Embarked=='C'),'Cabin']='F'

data_train.loc[(data_train.Cabin.isnull())&(data_train.Pclass==3)&(data_train.Embarked=='Q'),'Cabin']='F'


data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='C')&(data_test.Fare<=56.9292),'Cabin']='A'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='C')&(data_test.Fare>56.9292)&(data_test.Fare<=113.2750),'Cabin']='D'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='C')&(data_test.Fare>113.2750)&(data_test.Fare<=134.5000),'Cabin']='E'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='C')&(data_test.Fare>134.5000)&(data_test.Fare<=227.5250),'Cabin']='C'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='C')&(data_test.Fare>227.5250),'Cabin']='B'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='S')&(data_test.Fare<=35.5000),'Cabin']='T'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='S')&(data_test.Fare>35.5000)&(data_test.Fare<=77.9583),'Cabin']='D'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='S')&(data_test.Fare>77.9583)&(data_test.Fare<=79.6500),'Cabin']='E'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='S')&(data_test.Fare>79.6500)&(data_test.Fare<=81.8583),'Cabin']='A'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='S')&(data_test.Fare>81.8583)&(data_test.Fare<=211.3375),'Cabin']='B'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&
               (data_test.Embarked=='S')&(data_test.Fare>211.3375),'Cabin']='C'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==1)&(data_test.Embarked=='Q'),'Cabin']='C'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==2)&
               (data_test.Embarked=='S')&(data_test.Fare<=13.0000),'Cabin']=random.sample(['D','E'],1)
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==2)&
               (data_test.Embarked=='S')&(data_test.Fare>13.0000),'Cabin']='F'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==2)&(data_test.Embarked=='C'),'Cabin']='D'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==2)&(data_test.Embarked=='Q'),'Cabin']='E'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==3)&
               (data_test.Embarked=='S')&(data_test.Fare<=7.6500),'Cabin']='F'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==3)&
               (data_test.Embarked=='S')&(data_test.Fare>7.6500)&(data_test.Fare<=12.4750),'Cabin']='E'
data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==3)&
               (data_test.Embarked=='S')&(data_test.Fare>12.4750),'Cabin']='G'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==3)&(data_test.Embarked=='C'),'Cabin']='F'

data_test.loc[(data_test.Cabin.isnull())&(data_test.Pclass==3)&(data_test.Embarked=='Q'),'Cabin']='F'

print(data_test.Cabin.isnull().any(),'\n')

print(data_train.Cabin.isnull().any())
f,ax=plt.subplots(1,3,figsize=(23,6))

data_train.groupby(['Cabin','Survived'])['Survived'].count().plot.pie(autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Survived vs Dead by Cabin',color = 'r',fontsize=15)
ax[0].set_ylabel('')

sns.countplot('Survived',data=data_train,hue='Cabin',ax=ax[1])
ax[1].set_title('Survived vs Dead by Cabin',color = 'r',fontsize=15)

sns.barplot(x=data_train.groupby(['Cabin'])['Survived'].mean().index,
            y=data_train.groupby(['Cabin'])['Survived'].mean().values,ax=ax[2])
ax[2].set_title('Rate of the Survived by Cabin',color = 'r',fontsize=15)

plt.show()
Cabin=pd.concat([data_train[['Cabin','Embarked','Pclass','Fare']],data_test[['Cabin','Embarked','Pclass','Fare']]],axis=0)

data_train[data_train['Embarked'].isnull()]
f,ax=plt.subplots(1,2,figsize=(12,5))

sns.countplot('Survived',
              data=data_train.loc[(data_train.Pclass==1)&(data_train.Cabin=='B')&(data_train.Embarked.notnull())&(data_train.Sex=='female')],
              hue='Embarked',
              ax=ax[0])
ax[0].set_title('Survived',color = 'r',fontsize=15)

sns.barplot(x=data_train.loc[(data_train.Embarked=='C')|(data_train.Embarked=='S')].groupby(['Embarked'])['Survived'].mean().index,
            y=data_train.loc[(data_train.Embarked=='C')|(data_train.Embarked=='S')].groupby(['Embarked'])['Survived'].mean().values,
            ax=ax[1])
ax[1].set_title('Rate of the Survived by Embarked',color = 'r',fontsize=15)

plt.show()
data_train['Embarked'].fillna('C',inplace=True)

data_train.iloc[[61,829]]
print('Train columns with null values:\n',data_train.isnull().sum())
print("-"*40)
print('Test columns with null values:\n',data_test.isnull().sum())
data_train['Sex'].replace(['male','female'],[0,1],inplace=True)
data_train['Embarked'].replace(['C','Q','S'],[0,1,2],inplace=True)
data_train['Title'].replace(['Master','Miss','Mr','Mrs','Rare'],[0,1,2,3,4],inplace=True)
data_train['Cabin'].replace(['A','B','C','D','E','F','G','T'],[0,1,2,3,4,5,6,7],inplace=True)

data_train.loc[data_train['Age']<=16,'Age']=0
data_train.loc[(data_train['Age']>16)&(data_train['Age']<=32),'Age']=1
data_train.loc[(data_train['Age']>32)&(data_train['Age']<=48),'Age']=2
data_train.loc[(data_train['Age']>48)&(data_train['Age']<=64),'Age']=3
data_train.loc[data_train['Age']>64,'Age']=4

data_test['Sex'].replace(['male','female'],[0,1],inplace=True)
data_test['Embarked'].replace(['C','Q','S'],[0,1,2],inplace=True)
data_test['Title'].replace(['Master','Miss','Mr','Mrs','Rare'],[0,1,2,3,4],inplace=True)
data_test['Cabin'].replace(['A','B','C','D','E','F','G','T'],[0,1,2,3,4,5,6,7],inplace=True)

data_test.loc[data_test['Age']<=16,'Age']=0
data_test.loc[(data_test['Age']>16)&(data_test['Age']<=32),'Age']=1
data_test.loc[(data_test['Age']>32)&(data_test['Age']<=48),'Age']=2
data_test.loc[(data_test['Age']>48)&(data_test['Age']<=64),'Age']=3
data_test.loc[data_test['Age']>64,'Age']=4
data_train['Family_Size']=0
data_train['Family_Size']=data_train['Parch']+data_train['SibSp']

data_test['Family_Size']=0
data_test['Family_Size']=data_test['Parch']+data_test['SibSp']
sns.factorplot('Family_Size','Survived',data=data_train)
plt.title('Family_Size',color = 'r',fontsize=15)

plt.show()
data_train.loc[(data_train['Family_Size']>0)&(data_train['Family_Size']<4),'Family_Size']=1
data_train.loc[(data_train['Family_Size']>=4),'Family_Size']=2

data_test.loc[(data_test['Family_Size']>0)&(data_test['Family_Size']<4),'Family_Size']=1
data_test.loc[(data_test['Family_Size']>=4),'Family_Size']=2

f,ax=plt.subplots(1,2,figsize=(12,5))

sns.countplot('Family_Size',data=data_train,hue='Survived',ax=ax[0])
ax[0].set_title('Survived',color = 'r',fontsize=15)

sns.barplot(x=data_train.groupby(['Family_Size'])['Survived'].mean().index,
            y=data_train.groupby(['Family_Size'])['Survived'].mean().values,
            ax=ax[1])
ax[1].set_title('Rate of the Survived by Family_Size',color = 'r',fontsize=15)

plt.show()
Fare.groupby(['Pclass'])['Fare'].median()
data_train.loc[data_train['Fare']<=8.0500,'Fare']=0
data_train.loc[(data_train['Fare']>8.0500)&(data_train['Fare']<=15.0458),'Fare']=1
data_train.loc[(data_train['Fare']>15.0458)&(data_train['Fare']<=60.0000),'Fare']=2
data_train.loc[data_train['Fare']>60.0000,'Fare']=3

data_test.loc[data_test['Fare']<=8.0500,'Fare']=0
data_test.loc[(data_test['Fare']>8.0500)&(data_test['Fare']<=15.0458),'Fare']=1
data_test.loc[(data_test['Fare']>15.0458)&(data_test['Fare']<=60.0000),'Fare']=2
data_test.loc[data_test['Fare']>60.0000,'Fare']=3

f,ax=plt.subplots(1,2,figsize=(12,5))

sns.countplot('Fare',data=data_train,hue='Survived',ax=ax[0])
ax[0].set_title('Survived',color = 'r',fontsize=15)

sns.barplot(x=data_train.groupby(['Fare'])['Survived'].mean().index,
            y=data_train.groupby(['Fare'])['Survived'].mean().values,
            ax=ax[1])
ax[1].set_title('Rate of the Survived by Fare',color = 'r',fontsize=15)

plt.show()
data_train.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)
data_test.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)
f, ax = plt.subplots(figsize=(12, 12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data_train.corr(), cmap=cmap, vmax=.3, center=0,square=True,annot=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.pairplot(data_train,diag_kind="kde",hue="Survived")
plt.show()
y=data_train['Survived']

x=data_train.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
from sklearn import metrics #accuracy measure
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import GridSearchCV # # Grid Search Cross Validation
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.metrics import roc_curve # ROC Curve with logistic regression
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
mean=[] # List for mean of CV scores
accuracy=[] # List for CV score
std=[] # List for CV std

# Main function for models
def model(algorithm,x_train_,y_train_,x_test_,y_test_): 
    algorithm.fit(x_train_,y_train_)
    predicts=algorithm.predict(x_test_)
    prediction=pd.DataFrame(predicts)
    prob=algorithm.predict_proba(x_test_)[:,1]
    cross_val=cross_val_score(algorithm,x_train_,y_train_,cv=kfold)
    
    # Appending results to Lists 
    mean.append(cross_val.mean())
    std.append(cross_val.std())
    accuracy.append(cross_val)
    
    # Printing results  
    print(('{}'.format(algorithm)).split("(")[0].strip(),'\n') 
    print("CV std :",cross_val.std(),"\n")
    print("CV scores:",cross_val,"\n")
    print("CV mean:",cross_val.mean())
    
    # Plot for conf. matrix and roc curve
    fpr, tpr, thresholds = roc_curve(y_test_, prob)
    
    f,ax=plt.subplots(1,2,figsize=(11,4))
    
    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    # Plot for Confusion Matrix
    y_pred = cross_val_predict(algorithm,x,y,cv=10)
    sns.heatmap(confusion_matrix(y,y_pred),ax=ax[0],annot=True,fmt='2.0f')
    ax[0].set_title(('Confusion Matrix for {}'.format(algorithm)).split("(")[0].strip())
    
    plt.subplots_adjust(wspace=0.3)
    plt.close(0)
    plt.show()

# K-Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier

grids = {'n_neighbors': np.arange(1,50)}

grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=grids, cv=kfold) # Grid Search for best param.
grid.fit(x_train, y_train)

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(grid.best_params_),'\n') 
print("Best score: {}".format(grid.best_score_))
knn = KNeighborsClassifier(n_neighbors = grid.best_estimator_.n_neighbors)

model(knn,x_train,y_train,x_test,y_test)
# Support Vector Machines

from sklearn import svm 

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
grids = {'C': Cs, 'gamma' : gammas}

grid = GridSearchCV(estimator=svm.SVC(kernel='linear'), param_grid=grids, cv=kfold) # Grid Search for best param.
grid.fit(x_train, y_train)

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(grid.best_params_),'\n') 
print("Best score: {}".format(grid.best_score_))
svm = svm.SVC(kernel='linear',C=grid.best_estimator_.C,gamma=grid.best_estimator_.gamma,probability=True)

model(svm,x_train,y_train,x_test,y_test)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB 
nb = GaussianNB()
model(nb,x_train,y_train,x_test,y_test)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier 

grids={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}

grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=grids, cv=kfold) # Grid Search for best param.
grid.fit(x_train, y_train)

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(grid.best_params_),'\n') 
print("Best score: {}".format(grid.best_score_))
dtc = DecisionTreeClassifier(min_samples_split=grid.best_estimator_.min_samples_split, max_depth=grid.best_estimator_.max_depth)

model(dtc,x_train,y_train,x_test,y_test)
# Random Forest

from sklearn.ensemble import RandomForestClassifier 

grids={'n_estimators':range(100,500,100)}

grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grids, cv=kfold) # Grid Search for best param.
grid.fit(x_train, y_train)

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(grid.best_params_),'\n') 
print("Best score: {}".format(grid.best_score_))
rf = RandomForestClassifier(n_estimators=grid.best_estimator_.n_estimators)

model(rf,x_train,y_train,x_test,y_test)
# Logistic Regression

from sklearn.linear_model import LogisticRegression 

grids = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}

grid = GridSearchCV(estimator=LogisticRegression(), param_grid=grids, cv=kfold) # l1 lasso l2 ridge
grid.fit(x_train, y_train)

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(grid.best_params_),'\n') 
print("Best score: {}".format(grid.best_score_))
lr = LogisticRegression(C=grid.best_estimator_.C,penalty=grid.best_estimator_.penalty)

model(lr,x_train,y_train,x_test,y_test)
classifiers=['KNN','Svm','Naive Bayes','Decision Tree','Random Forest','Logistic Regression']

models=pd.DataFrame({'CV mean':mean,'Std':std},index=classifiers)       
print(models)
f, ax = plt.subplots(figsize=(16, 7))

sns.boxplot(x=models.index, y=accuracy)
plt.xticks(rotation=45)
plt.title('Models',color = 'r',fontsize=15)

plt.show()
coefficients=pd.DataFrame({'Features':data_test.columns,'Coefficients':dtc.feature_importances_})       

plt.figure(figsize=(15,5))
sns.barplot(x=coefficients['Features'],y=coefficients['Coefficients'])
plt.xticks(rotation=45)
plt.title('Titles',color = 'r',fontsize=15)

plt.show()
submission = pd.DataFrame({"PassengerId": pd.read_csv("../input/test.csv")["PassengerId"],"Survived": dtc.predict(data_test)})

submission.to_csv('titanic.csv', index=False)