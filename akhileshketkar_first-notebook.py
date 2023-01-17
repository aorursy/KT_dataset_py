import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/traincsv/train.csv')

df.drop(['PassengerId'],axis=1,inplace=True)

df
df['Ticket_extension'] = 0

for i in list(df['Ticket'].values):

    df.loc[df['Ticket'] == i,['Ticket_extension']] = i.split(' ')[0]
name = [i.split(',')[1].strip() for i in list(df['Name'].values)]

title = []

for i in name:

    extension = i.split('.')[0]

    if extension == 'Mr' or extension == 'Mrs' or extension == 'Miss' or extension == 'Master':

        title.append(extension)

    else:

        title.append('Other')       

df['Title'] = title
df['Title'].unique()
g = sns.factorplot(x='Title',y='Survived',data=df)

g.set_ylabels('Survived probability')
df['Age'].hist(bins=20)
df.loc[df['Ticket'] == '364516']
df['Age'].isna().mean()
new_df = df.copy()

df.loc[df['Age'].isna(),'Title'].value_counts()
Mr_mean = round(np.nanmean(new_df.loc[df['Title'] == 'Mr','Age'].values))

Miss_mean = round(np.nanmean(new_df.loc[df['Title'] == 'Miss','Age'].values))

Mrs_mean = round(np.nanmean(new_df.loc[df['Title'] == 'Mrs','Age'].values))

Master_mean = round(np.nanmean(new_df.loc[df['Title'] == 'Master','Age'].values))

Other_mean = round(np.nanmean(new_df.loc[df['Title'] == 'Miss','Age'].values))
for i,y in enumerate(df['Age'].values):  

    if np.isnan(df['Age'][i]):

        y = df['Title'][i]

        if y == 'Mr':

            new_df.loc[i,'Age'] = Mr_mean

        if y == 'Miss':

            new_df.loc[i,'Age'] = Miss_mean 

        if y == 'Mrs':

            new_df.loc[i,'Age'] = Mrs_mean 

        if y == 'Master':

            new_df.loc[i,'Age'] = Master_mean 

        if y == 'Other':

            new_df.loc[i,'Age'] = Other_mean 
new_df['Age type'] = 0

new_df.loc[new_df['Age'] < 16,'Age type'] = 'Child'

new_df.loc[new_df['Age'].between(16,60),'Age type'] = 'Adult'

new_df.loc[new_df['Age'] > 60,'Age type'] = 'Old'
g = sns.factorplot(x='Age type',y='Survived',data=new_df,kind='bar')

g = g.set_ylabels("Survival probability")
df = new_df.copy()
fig = plt.subplots(figsize=(10,6))

sns.countplot(x='SibSp',hue='Survived',data=df)

plt.legend(loc = 'right')
df.loc[df['SibSp']==8]
sns.countplot(x='Parch',hue='Survived',data=df)
df.loc[df['Parch']>5]
df.loc[df['Ticket_extension'] == 'CA']
Sibsp = df['SibSp'].values

Parch = df['Parch'].values

family_size = np.add(Sibsp,Parch)
df['Family size'] = family_size
g = sns.factorplot(x='Family size',y='Survived',data=df,kind='bar')

g = g.set_ylabels("Survival probability")
df['Family remark'] = 0

df.loc[df['Family size'] == 0 ,'Family remark'] = 'No Family'

df.loc[df['Family size'].between(1,3),'Family remark'] = 'Small Family'

df.loc[df['Family size'].between(4,10),'Family remark'] = 'Large Family'
df.loc[df['Family size'] > 6]
sns.factorplot(x='Family remark',y='Survived',data=df)
sns.distplot(df['Fare'])
df.loc[df['Fare'] > 300,'Cabin']
df.loc[df['Cabin'] == 'B51 B53 B55','Fare'].value_counts()
#Droping this values beacause cabin is same for lower value

df.drop([679,258,737],axis=0,inplace=True)
df.loc[df['Fare'] > 250]
df.loc[df['Fare'].between(0,75),'Survived'].value_counts()
df.loc[df['Fare'].between(75,300),'Survived'].value_counts()
df['Fare_range'] = 0

df.loc[df['Fare'].between(0,75),'Fare_range'] = 'Low Fare'

df.loc[df['Fare'].between(75,300),'Fare_range'] = 'High Fare'
g = sns.factorplot(x='Fare_range',y='Survived',kind='bar',data=df)

g.set_ylabels('Survival probability')
g =sns.factorplot(x='Fare_range',y='Survived',hue='Family remark',kind='bar',data=df)

g.set_ylabels('Survival probability')
new_df = df.copy()
df['Cabin'].isna().mean()
#Finding realtion between NA values

df.loc[df['Cabin'].isna(),'Fare_range'].value_counts()
df['Cabin remark'] = 0

df.loc[df['Cabin'].isna(),'Cabin remark'] = 'No Cabin'

df.loc[~df['Cabin'].isna(),'Cabin remark'] = 'With Cabin'
new_df = df.copy()

df.loc[new_df['Embarked'].isna(),'Ticket_extension']
df.loc[df['Ticket_extension']=='113572']
df.drop([61,829],axis=0,inplace=True)
new_df['Cabin'].fillna('No cabin',inplace=True)
df.head()
g =sns.factorplot(x='Cabin remark',y='Survived',data=df)

g.set_ylabels('Survival probability')
g =sns.factorplot(x='Family size',y='Survived',data=df)

g.set_ylabels('Survival probability')
g =sns.factorplot(x='Age type',y='Survived',data=df)

g.set_ylabels('Survival probability')
g =sns.factorplot(x='Family remark',y='Survived',hue='Fare_range',kind='bar',data=df)

g.set_ylabels('Survival probability')
df.shape
g =sns.factorplot(x='Fare_range',y='Survived',data=df)

g.set_ylabels('Survival probability')
df2 = df[['Sex','Age type','Cabin remark','Fare_range','Title']]

Pclass = pd.get_dummies(df['Pclass'])

Family_size = pd.get_dummies(df['Family size'])

X1 = pd.get_dummies(df2)
X = pd.concat([Pclass,Family_size,X1],axis=1)

X = X.reset_index(drop=True)

X
X_data = X.values
Y = df['Survived']

Y = Y.reset_index(drop=True)

Y
Y_data = Y.values