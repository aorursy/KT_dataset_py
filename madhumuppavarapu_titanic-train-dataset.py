import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px 
train_df=pd.read_csv('../input/titanicdataset-traincsv/train.csv')

train_df
train_df.head()
train_df.info()
train_df.describe()
train_df.isnull().sum()
train_df.isnull()
sns.heatmap(train_df.isnull())
print("no. of males in the titanic:",train_df['Sex'].value_counts()['male'])

print("no. of females in the titanic:",train_df['Sex'].value_counts()['female'])

train_df['Survived'].value_counts()[train_df['Sex']=='male']

train_df['Survived'].value_counts()[train_df['Sex']=='female']
plt.subplot(1,2,1)

sns.countplot(x='Sex',data=train_df)

plt.subplot(1,2,2)

sns.countplot(data=train_df,x='Survived')
sns.countplot(x='Survived',data=train_df,palette='rainbow',hue='Sex')
sns.countplot(x='Survived',data=train_df,palette='rainbow',hue='Pclass')
train_df['Fare']//=100
sns.countplot(data=train_df,x='Survived',hue=train_df['Fare'],palette='rainbow')
sns.set_style('whitegrid')

sns.distplot(train_df['Age'],kde=False,bins=20,color='g')
train_df['Survived'].value_counts()[1]
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(ascending=False,by='SibSp')
sns.countplot(data=train_df,x='Survived',hue='SibSp',palette='rainbow')

plt.legend()
x=train_df['SibSp']

y=train_df['Survived']

fig,Axes=plt.subplots()

plt.suptitle('SibSp vs Survived')

plt.subplot(1,3,1)

plt.scatter(x,y,marker='*',color='r',linewidth=5,s=25,edgecolor='g')

Axes.set_title('using scatterplot')

plt.subplot(1,3,2)

plt.xlabel('SibSp')

plt.ylabel('Survived')

Axes.set_title('using plot ')

plt.plot(x,y,'g*',linestyle='dashdot',linewidth=2,markersize=10)

plt.subplot(1,3,3)

plt.bar(x,y,align='center',color='black')

Axes.set_title('using bar')

plt.xlabel('SibSp')

plt.ylabel('Survived')

train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Sex',ascending=False)
sns.boxplot(x='SibSp',y='Age',data=train_df)
def fill_age(cols):

    SibSp = cols[0]

    Age =cols[1]

    if pd.isnull(Age):

        if SibSp==0:

            return 29

        elif SibSp==1:

            return 30

        elif SibSp==2:

            return 23

        elif SibSp==3:

            return 10

        elif SibSp==4:

            return 7

        elif SibSp==5:

            return 11

        else:

            return train_df.fillna('ffill')

    else:

        return Age

    
train_df['Age']=train_df[["Age","SibSp"]].apply(fill_age,axis=1)
sns.heatmap(train_df.isnull())
train_df.isnull().sum()
train_df['Embarked'].fillna('bfill',inplace=True)
train_df.isnull().sum()
train_df.info()
# we can represent the given values except Name,Sex,Embarked ,Ticket

# so we will convert object datatype into categorical values if possible or we will drop the unnecessary columns

pd.get_dummies(train_df)
# so we will create a copy of train_df and proceed accordingly

train_copy=train_df.copy()

train_copy
# now we will drop name and ticket columns because they can't be converted into valid categorical columns

train_copy.drop(['Name','Ticket'],inplace=True,axis=1)
train_copy
# so we will convert the Embarked and Sex to categorical values using get_dummies()

Sex_category=pd.get_dummies(train_copy['Sex'],drop_first=True)

Embarked_category=pd.get_dummies(train_copy['Embarked'],drop_first=True)
# drop Sex and Embarked

train_copy.drop(['Sex','Embarked'],axis=1,inplace=True)
# now we will add Sex_category and Embarked_category into the train_copy DataFrame

train=pd.concat([train_copy,Sex_category,Embarked_category],axis=1)
train.head()
train.drop(['bfill'],axis=1,inplace=True)
train.info()
sns.rugplot(train['Age'].isnull())
sns.jointplot(data=train,x=train['Survived'],y=train['Pclass'],kind='kde')
sns.pairplot(train)
sns.distplot(train[['Survived','Pclass']],kde=True,bins=10)
sns.jointplot(x=train['male'],y=train['Pclass'],kind='kde')
sns.heatmap(train.corr())
sns.boxplot(x='male',y='Pclass',data=train,color='k')

sns.boxenplot(x='male',y='Pclass',data=train,color='g')
sns.swarmplot(x='male',y='Pclass',data=train,color='k')

sns.violinplot(x='male',y='Pclass',data=train,color='g')

sns.stripplot(x='male',y='Pclass',data=train,color='r')
sns.stripplot(x='Survived',y='SibSp',data=train,color='b')

sns.swarmplot(x='Survived',y='SibSp',data=train,color='k')

sns.violinplot(x='Survived',y='SibSp',data=train,palette='rainbow')

sns.boxenplot(data=train,x='Survived',y='SibSp',color='m')

sns.barplot(data=train,y='SibSp',x='Survived',color='y')

sns.boxplot(x='Survived',y='SibSp',data=train,palette='dark')

sns.countplot(data=train,y='SibSp',color='red')
sns.factorplot(x='Pclass',y='SibSp',data=train)
train.head(10)