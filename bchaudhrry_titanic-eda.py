import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
data = pd.read_csv('../input/titanic_data.csv')

dftitanic = data.copy()
data_comp = pd.read_csv('../input/competition.csv')

dfcomp = data_comp.copy()
dftitanic.shape
dfcomp.shape
dftitanic.head()
dfcomp.head()
dftitanic.dtypes
dfcomp.dtypes
dftitanic.describe()
dftitanic.describe(include=["O"])
dftitanic.describe(include='all')
dfcomp.describe()
dfcomp.describe(include='all')
dfcomp.describe(include=['O'])
dftitanic.isnull().sum()
dfcomp.isnull().sum()
sns.countplot(x='Survived', hue='Survived', data=dftitanic);
sns.countplot(x='Sex', data=dftitanic);
sns.countplot(x='Sex', hue='Survived', data=dftitanic);
sns.countplot(x='Sex', data=dfcomp);
plt.figure(figsize=(10,6))

sns.countplot(x='Pclass', data=dftitanic);
sns.countplot(x='Pclass', data=dfcomp);
sns.countplot(x='Pclass', hue='Survived', data=dftitanic);
sns.countplot(x='Sex', hue='Pclass', data=dftitanic);
sns.catplot(x='Sex', col='Pclass', data=dftitanic, kind='count');
sns.catplot(x='Sex', col='Pclass', hue='Survived', data=dftitanic, kind='count');
sns.factorplot(x='Sex', col='Pclass', kind='count', data=dfcomp);
dftitanic['Name'].head(10)
import re 

a=' Braund, Mr.Owen Harris'

re.search(' ([A-Z][a-z]+)\.', a).group(1)
re.search(' ([A-Z][a-z]+)\.',a).group(0)
dftitanic['Title'] = dftitanic['Name'].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

dftitanic.head() #apply is pandas function
dftitanic['Title'].value_counts()
dfcomp['Titles']=dfcomp['Name'].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

dfcomp['Titles'].head()
dfcomp['Titles'].value_counts()
dftitanic['Title'] = dftitanic['Title'].replace('Mlle', 'Miss')
dftitanic['Title'] = dftitanic['Title'].replace('Mme', 'Mrs')
dftitanic.loc[(~dftitanic['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master'])), 'Title'] = 'Rare Title'

#not(~)
dftitanic['Title'].unique()
sns.countplot(data=dftitanic,x='Title', hue='Survived');
dfcomp['Titles']=dfcomp['Titles'].replace('Mme','Mrs')
dfcomp.loc[(~dfcomp['Titles'].isin(['Mr','Mrs','Miss','Master'])), 'Titles'] = 'Rare Title'
dfcomp['Titles'].unique()
sns.countplot(data=dfcomp, x='Titles');
dftitanic['Fsize'] = dftitanic['SibSp'] + dftitanic['Parch']+1
# Write the code here

sns.countplot(x='Fsize', hue='Survived', data=dftitanic);
dftitanic.groupby('Fsize')['Survived'].value_counts(normalize=True).reset_index(name='perc')
temp = dftitanic.groupby('Fsize')['Survived'].value_counts(normalize=True).reset_index(name='Perc')



plt.figure(figsize=(15,6));

sns.barplot(data=temp,x='Fsize', y='Perc', hue='Survived', dodge=True, alpha=0.5);
dftitanic['Ticket'].value_counts().head()
dftitanic['Ticket'].value_counts().reset_index().head()
temp = dftitanic['Ticket'].value_counts().reset_index(name='Tsize')

temp
dftitanic = dftitanic.merge(temp, left_on='Ticket', right_on='index',how='inner').drop('index', axis=1)

dftitanic.head()
sns.countplot(x='Tsize', hue='Survived', data=dftitanic);
temp = dftitanic.groupby('Tsize')['Survived'].value_counts(normalize=True).reset_index(name='Perc')
sns.barplot(x='Tsize', y='Perc', hue='Survived', data=temp, dodge=True);
dftitanic['Group'] = dftitanic[['Tsize', 'Fsize']].max(axis=1)



plt.figure(figsize=(15,6));

sns.countplot(x='Group', hue='Survived', data=dftitanic);
dftitanic['GrpSize'] = ''

dftitanic.loc[dftitanic['Group']==1, 'GrpSize'] = dftitanic.loc[dftitanic['Group']==1, 'GrpSize'].replace('', 'solo')

dftitanic.loc[dftitanic['Group']==2, 'GrpSize'] = dftitanic.loc[dftitanic['Group']==2, 'GrpSize'].replace('', 'couple')

dftitanic.loc[(dftitanic['Group']<=4) & (dftitanic['Group']>=3), 'GrpSize'] = dftitanic.loc[(dftitanic['Group']<=4) & (dftitanic['Group']>=3), 'GrpSize'].replace('', 'group')

dftitanic.loc[dftitanic['Group']>4, 'GrpSize'] = dftitanic.loc[dftitanic['Group']>4, 'GrpSize'].replace('', 'large group')

dftitanic.head()
plt.figure(figsize=(15,6));

sns.countplot(x='GrpSize', order=['solo', 'couple', 'group', 'large group'], hue='Survived', data=dftitanic);
dfcomp.columns
dfcomp['FSize'] = dfcomp['SibSp'] + dfcomp['Parch']+1
temp=dfcomp['Ticket'].value_counts().reset_index(name='Tsize')
dfcomp=dfcomp.merge(temp, left_on='Ticket', right_on='index',how='inner').drop('index', axis=1)
dfcomp['Group'] = dfcomp[['Tsize', 'FSize']].max(axis=1)

plt.figure(figsize=(15,6));

sns.countplot(x='Group', hue='Survived', data=dftitanic);
dfcomp['GrpSize'] = ''

dfcomp.loc[dfcomp['Group']==1, 'GrpSize'] = dfcomp.loc[dfcomp['Group']==1, 'GrpSize'].replace('', 'solo')

dfcomp.loc[dfcomp['Group']==2, 'GrpSize'] = dfcomp.loc[dfcomp['Group']==2, 'GrpSize'].replace('', 'couple')

dfcomp.loc[(dfcomp['Group']<=4) & (dfcomp['Group']>=3), 'GrpSize'] = dfcomp.loc[(dfcomp['Group']<=4) & (dfcomp['Group']>=3), 'GrpSize'].replace('', 'group')

dfcomp.loc[dfcomp['Group']>4, 'GrpSize'] = dfcomp.loc[dfcomp['Group']>4, 'GrpSize'].replace('', 'large group')

dfcomp.head()
dftitanic['Fare'].isnull().sum()
plt.figure(figsize=(15,6))

sns.distplot(dftitanic['Fare']);
dftitanic[dftitanic['Fare'] < 0]
dftitanic[dftitanic['Fare'] == 0]
dftitanic.loc[(dftitanic['Fare'] == 0) & (dftitanic['Pclass'] == 1), 'Fare'] = dftitanic[dftitanic['Pclass'] == 1]['Fare'].median()

dftitanic.loc[(dftitanic['Fare'] == 0) & (dftitanic['Pclass'] == 2), 'Fare'] = dftitanic[dftitanic['Pclass'] == 2]['Fare'].median()

dftitanic.loc[(dftitanic['Fare'] == 0) & (dftitanic['Pclass'] == 3), 'Fare'] = dftitanic[dftitanic['Pclass'] == 3]['Fare'].median()
dftitanic[dftitanic['Fare']==0]
dftitanic['FareCat'] = ''

dftitanic.loc[dftitanic['Fare']<=10, 'FareCat'] = '0-10'

dftitanic.loc[(dftitanic['Fare']>10) & (dftitanic['Fare']<=25), 'FareCat'] = '10-25'

dftitanic.loc[(dftitanic['Fare']>25) & (dftitanic['Fare']<=40), 'FareCat'] = '25-40'

dftitanic.loc[(dftitanic['Fare']>40) & (dftitanic['Fare']<=70), 'FareCat'] = '40-70'

dftitanic.loc[(dftitanic['Fare']>70) & (dftitanic['Fare']<=100), 'FareCat'] = '70-100'

dftitanic.loc[dftitanic['Fare']>100, 'FareCat'] = '100+'

dftitanic[['Fare', 'FareCat']].head()
plt.subplots(figsize=(15,6))

sns.countplot(x='FareCat', order=['0-10', '10-25', '25-40', '40-70', '70-100', '100+'], hue='Survived', data=dftitanic);
dfcomp[dfcomp['Fare']==0]
dfcomp.loc[(dfcomp['Fare'] == 0) & (dfcomp['Pclass'] == 1), 'Fare'] = dfcomp[dfcomp['Pclass'] == 1]['Fare'].median()

dfcomp.loc[(dfcomp['Fare'] == 0) & (dfcomp['Pclass'] == 2), 'Fare'] = dfcomp[dfcomp['Pclass'] == 2]['Fare'].median()

dfcomp.loc[(dfcomp['Fare'] == 0) & (dfcomp['Pclass'] == 3), 'Fare'] = dfcomp[dfcomp['Pclass'] == 3]['Fare'].median()

dfcomp['FareCat'] = ''

dfcomp.loc[dfcomp['Fare']<=10, 'FareCat'] = '0-10'

dfcomp.loc[(dfcomp['Fare']>10) & (dfcomp['Fare']<=25), 'FareCat'] = '10-25'

dfcomp.loc[(dfcomp['Fare']>25) & (dfcomp['Fare']<=40), 'FareCat'] = '25-40'

dfcomp.loc[(dfcomp['Fare']>40) & (dfcomp['Fare']<=70), 'FareCat'] = '40-70'

dfcomp.loc[(dfcomp['Fare']>70) & (dfcomp['Fare']<=100), 'FareCat'] = '70-100'

dfcomp.loc[dfcomp['Fare']>100, 'FareCat'] = '100+'

dfcomp[['Fare', 'FareCat']].head()
plt.subplots(figsize=(15,6))

sns.countplot(x='Embarked', hue='Survived', data=dftitanic)
dftitanic.head()
dfcomp.head()
sns.kdeplot(dftitanic[dftitanic['Survived'] == 0]['Age'].dropna(), shade=True,label="Not survived");

sns.kdeplot(dftitanic[dftitanic['Survived'] == 1]['Age'].dropna(), shade=True,label="Survived");
temp = dftitanic[dftitanic['Age'].isnull() == False]
sns.set_context('poster')

sns.factorplot(kind='box', x='Age', col='Title', row='Pclass', data=temp);
for t in dftitanic['Title'].unique():

    for p in dftitanic['Pclass'].unique():

        dftitanic.loc[(dftitanic['Title'] == t) & (dftitanic['Pclass'] == p) & (dftitanic['Age'].isnull()), 'Age'] = dftitanic.loc[(dftitanic['Title'] == t) & (dftitanic['Pclass'] == p), 'Age'].median()
plt.subplots(figsize=(10,6))

sns.kdeplot(dftitanic[dftitanic['Survived'] == 0]['Age'], shade=True,label='Not Survived');

sns.kdeplot(dftitanic[dftitanic['Survived'] == 1]['Age'], shade=True,label='Survived');
dftitanic['Age'].isnull().sum()
dftitanic['AgeCat']=''

dftitanic.loc[ dftitanic['Age'] <= 16, 'AgeCat'] = '0-16'

dftitanic.loc[(dftitanic['Age'] > 16) & (dftitanic['Age'] <= 32), 'AgeCat'] = '16-32'

dftitanic.loc[(dftitanic['Age'] > 32) & (dftitanic['Age'] <= 48), 'AgeCat'] = '32-48'

dftitanic.loc[(dftitanic['Age'] > 48) & (dftitanic['Age'] <= 64), 'AgeCat'] = '48-64'

dftitanic.loc[ dftitanic['Age'] > 64, 'AgeCat']= '64+'
sns.countplot( x= "AgeCat", hue= "Survived", data= dftitanic);
for t in dfcomp['Titles'].unique():

    for p in dfcomp['Pclass'].unique():

        dfcomp.loc[(dfcomp['Titles']==t) & (dfcomp['Pclass']== p) & (dfcomp['Age'].isnull()),'Age']= dfcomp.loc[(dfcomp['Titles']==t) & (dfcomp['Pclass']==p), 'Age'].median()



dfcomp['Age'].isnull().sum()
dfcomp.loc[(dfcomp['Age'].isnull()),'Pclass':'Titles'] 
dfcomp.loc[(dfcomp['Age'].isnull()) & (dfcomp['Pclass'] == 3) & (dfcomp['Titles'] == 'Rare Title'), 'Age'] = dfcomp.loc[(dfcomp['Titles'] == 'Rare Title') & (dfcomp['Pclass'] == 3), 'Age'].median()
dfcomp['Fare'].isnull().sum()
dfcomp.loc[(dfcomp['Fare'].isnull()),'Pclass']
dfcomp.loc[(dfcomp['Fare'].isnull()) & (dfcomp['Pclass']==3),'Fare']=dfcomp[dfcomp['Pclass']==3]['Fare'].median()

dfcomp.head()
dfcomp.loc[dfcomp['Fare']<=10, 'FareCat'] = '0-10'
dfcomp['AgeCat']=''

dfcomp.loc[dfcomp['Age']<=16,'AgeCat']='0-16'

dfcomp.loc[(dfcomp['Age']>16) & (dfcomp['Age']<=32),'AgeCat']='16-32'

dfcomp.loc[(dfcomp['Age']>32) & (dfcomp['Age']<=48),'AgeCat']='32-48'

dfcomp.loc[(dfcomp['Age']>48) & (dfcomp['Age']<=64),'AgeCat']='48-64'

dfcomp.loc[dfcomp['Age']>64,'AgeCat']='64+'

dfcomp.head()
dftitanic.describe(include='O')
# code to check the nulls in Embarked

dftitanic['Embarked'].isnull().sum()
dftitanic.loc[(dftitanic['Embarked'].isnull()),'Embarked']='S'
dfcomp['Embarked'].isnull().sum()
dftitanic['CabinType'] = dftitanic['Cabin'].str[0]
dftitanic['CabinType'].head()
plt.figure(figsize=(15,6))

sns.countplot(x='CabinType', hue='Survived', data=dftitanic);
dftitanic.groupby(['CabinType', 'Pclass'])['Pclass'].count()
dftitanic.drop('Cabin',axis=1,inplace=True)
dfcomp.drop('Cabin', axis=1,inplace=True)
dftitanic.isnull().sum()
dfcomp.isnull().sum()
sns.set_context('poster')
sns.set_context('poster')

plt.figure(figsize=(15,6))

cor = dftitanic.drop('PassengerId',axis=1).corr()

sns.heatmap(cor, annot=True, fmt='.1g');
drop_features = ['Name','Age','Fare','Ticket','Fsize','Tsize','Group']
dftitanic.drop(drop_features,axis=1,inplace=True)
sns.set()

cor=dfcomp.drop('PassengerId',axis=1).corr()

sns.heatmap(cor, annot=True, fmt='.1g');
drop_features = ['Name','Age','Fare','Ticket','FSize','Tsize','Group']
dfcomp.drop(drop_features,axis=1,inplace=True)
dftitanic.to_csv('titanic_clean.csv',index=False)
dfcomp.to_csv('competiton_clean.csv',index=False)