import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fancyimpute import MICE#Imputation

import missingno as msno #Missing Value Viualization

import matplotlib.pyplot as plt #Graphs

import seaborn as sns #Graphs

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
print(train_df.dtypes)

print("===================================================================================")

print(test_df.dtypes)
obj_to_cat = ["Name","Sex","Ticket","Cabin","Embarked"]

train_df[obj_to_cat].head()
train_df.isnull().any()

train_df.drop_duplicates()

summary_df = train_df.describe().transpose()

summary_df['count%']=(summary_df['count']/summary_df['count'].max())*100

summary_df
msno.matrix(train_df.sample(891))
#train_df.drop(['PassengerID'],axis=1,inplace=True)

train_df['Pclass'] = train_df['Pclass'].astype('category')

train_df['Sex'] = train_df['Sex'].astype('category')

train_df['Ticket'] = train_df['Ticket'].astype('category')



train_df.dtypes
train_df["Name"] = train_df["Name"].str.extract("([A-Za-z]+)\.",expand=False).astype('category')
passenger = ["Name","Sex","Age","SibSp","Parch"]

train_df[passenger].groupby('Name').describe().transpose()
train_df[['Name','Age']].isnull().any()

train_df.groupby('Name')['Age'].mean()

def age_na_fill(x):

    x['Age'] = train_df.groupby('Name')['Age'].mean()[x['Name']]

    return x['Age']    

train_df.loc[train_df['Age'].isnull(),["Age"]] = train_df.loc[train_df['Age'].isnull(),["Age","Name"]].apply(age_na_fill,axis=1)

train_df['Age'].isnull().any()
#pd.set_option('display.max_rows', None)

ship = ["Ticket","Fare","Cabin","Embarked","Pclass","PassengerId"]

train_df[ship].groupby("Ticket").count()
train_df["Ticket_cat"]=train_df['Ticket'].str.extract('([a-zA-z])',expand=False)

train_df["Ticket_cat"] = train_df["Ticket_cat"].fillna("N")#numerical

train_df["Ticket_cat"] = train_df["Ticket_cat"].astype('category')
train_df['Cabin_cat'] = train_df['Cabin'].str.extract('([a-zA-z])',expand=True)

train_df['Cabin_cat'] = train_df['Cabin_cat'].fillna('U')#unknown

train_df['Cabin_cat'] = train_df['Cabin_cat'].astype('category')
mode_val = train_df['Embarked'].mode().to_string(index =False)

train_df['Embarked']=train_df['Embarked'].fillna(value = "S")

train_df['Embarked'] = pd.Categorical(train_df['Embarked'],categories=["C","Q","S"])

train_df.drop(['Cabin'],inplace=True,axis=1)

train_df.dtypes

train_df.isnull().any()#After imputation DataFrame is free from NA
train_df.head()

train_df[passenger]

train_df['family_size']=train_df['Parch'] + train_df['SibSp']
train_df['Alone'] = 0

train_df.loc[train_df['family_size'] > 0,'Alone'] = 0 #has family

train_df.loc[train_df['family_size'] == 0,'Alone'] = 1

train_df.head()
train_df.drop(['Ticket','PassengerId'],axis=1,inplace=True)
train_df.head()
plt.figure(figsize=(10,10))

plt.title('Pearson Correlation of Features', size=15)

sns.heatmap(train_df.corr(),cmap=plt.cm.RdBu,annot=True)

plt.show()
sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Embarked")

plt.show()
plt.figure(figsize=(20,20))

sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Cabin_cat")

plt.show()
plt.figure(figsize=(20,20))

sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Ticket_cat")

plt.show()
plt.figure(figsize=(20,20))

sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Name")

plt.show()
plt.figure(figsize=(20,20))

sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Pclass")

plt.show()
plt.figure(figsize=(20,20))

sns.factorplot('Survived','Age',hue='Alone',data =train_df,kind="bar",col="Pclass")

plt.show()