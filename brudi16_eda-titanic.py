import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #plots

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_columns', 30)#max columns to 30
def plot_hue_survived(data,valor,hue,bins=None):

    if(bins==None):

        g = sns.catplot(valor,hue=hue,data=data, kind='count')

        g.set_axis_labels(valor, 'Nº passengers')

    else:

        cuts=int(max(data[valor])/bins)

        data['Tmp'] = data[valor].map(lambda fare: cuts * (fare // cuts))

        g = sns.catplot('Tmp',hue=hue,data=data, kind='count')

        g.set_axis_labels(valor, 'Nº passengers')

    

def percentaje_of_(data,dato1,dato2,bins=None):

    if(bins!=None):

        data['Tmp']=pd.cut(data[dato1], bins)

        dato1='Tmp'

    group_by=data[[dato1, dato2]].groupby([dato1], as_index=False).count()

    group_by_2=group_by[dato2].tolist()

    percent=[(x / sum(group_by_2)*100) for x in group_by_2]

    d = {dato1: group_by[dato1].tolist(), '% Percent.': percent}

    dataframe=pd.DataFrame(data=d)

    return dataframe

#lectura del dataset train y test

train_df = pd.read_csv('./train.csv')

test_df = pd.read_csv('./test.csv')

print(train_df.columns.values)

print('_'*40)
train_df["Survived"]=train_df["Survived"].astype('bool')
train_df["Pclass"]=train_df["Pclass"].astype('category')

train_df["Sex"]=train_df["Sex"].astype('category')

train_df["Cabin"]=train_df["Cabin"].astype('category')

train_df["Ticket"]=train_df["Ticket"].astype('category')

train_df["Embarked"]=train_df["Embarked"].astype('category')

train_df.info()

train_df.describe(include='all')
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
plot_hue_survived(train_df,"Pclass","Survived")
dataframe=percentaje_of_(train_df,'Pclass','PassengerId')

dataframe
g = sns.catplot('Pclass',data=train_df, kind='count')
train_df['Name'].describe(include='all')
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
plot_hue_survived(train_df,"Sex","Survived")
dataframe=percentaje_of_(train_df,'Sex','PassengerId')

dataframe
g = sns.catplot('Sex',data=train_df, kind='count')
train_df['Tmp']=pd.cut(train_df['Age'], 10)

train_df[['Tmp', 'Survived']].groupby(['Tmp'], as_index=False).mean()
plot_hue_survived(train_df,"Age","Survived",10)
dataframe=percentaje_of_(train_df,'Age','PassengerId',10)

dataframe
train_df['Age'].hist(bins=10,grid=False);
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
plot_hue_survived(train_df,"SibSp","Survived")
dataframe=percentaje_of_(train_df,'SibSp','PassengerId')

dataframe
g = sns.catplot('SibSp',data=train_df, kind='count')
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
plot_hue_survived(train_df,"Parch","Survived")
dataframe=percentaje_of_(train_df,'Parch','PassengerId')

dataframe
g = sns.catplot('Parch',data=train_df, kind='count')
train_df['Ticket'].describe(include='all')
train_df['Tmp']=pd.cut(train_df['Fare'], 10)

train_df[['Tmp', 'Survived']].groupby(['Tmp'], as_index=False).mean()
plot_hue_survived(train_df,"Fare","Survived",10)
dataframe=percentaje_of_(train_df,'Fare','PassengerId',10)

dataframe
train_df['Fare'].hist(bins=10,grid=False);
train_df['Cabin'].describe(include='all')
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
plot_hue_survived(train_df,"Embarked","Survived")
dataframe=percentaje_of_(train_df,'Embarked','PassengerId')

dataframe
g = sns.catplot('Embarked',data=train_df, kind='count')
dataframe=percentaje_of_(train_df,'Survived','PassengerId')

dataframe
g = sns.catplot('Survived',data=train_df, kind='count')