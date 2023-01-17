import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot

from wordcloud import WordCloud

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly import tools

from datetime import date

import seaborn as sns

import random 

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/titanic/train.csv")

train_df.head()
test_df=pd.read_csv("../input/titanic/test.csv")

test_df.head()
print('---'*40)

print("The number of Features in  train dataset :",train_df.shape[1])

print("The number of Rows in Train dataset :",train_df.shape[0])

print('---'*40)

print('-----Test Dataset------------------------------')

print("The number of Features in  test dataset :",test_df.shape[1])

print("The number of Rows in  Test dataset :",test_df.shape[0])
def type_features(data):

    categorical_features = data.select_dtypes(include = ["object"]).columns

    numerical_features = data.select_dtypes(exclude = ["object"]).columns

    print( "categorical_features :",categorical_features)

    print('-----'*40)

    print("numerical_features:",numerical_features)
print('Train_dataset')

print('````'*40)

type_features(train_df)
print('Test_dataset')

print('````'*40)

type_features(test_df)
def missingdata(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    ms= ms[ms["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='90')

    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return ms
missingdata(train_df)
print('------------------------------Test_Dataset--------------------------------------------')

missingdata(test_df)
f,ax=plt.subplots(1,2,figsize=(8,5))

train_df.Survived.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Distribution of target variable')

sns.countplot('Survived',data=train_df,ax=ax[1])

ax[1].set_title('Count of Survived VS Not Survived Passengers')

plt.show() 
def group_by(df,t1='',t2=''):

    a1=df.groupby([t1,t2])[t2].count()

    return a1
def plot_re(df,t1='',t2=''):

    f,ax=plt.subplots(1,2,figsize=(15,5))

    df[[t1,t2]].groupby([t1]).count().plot.bar(ax=ax[0])

    ax[0].set_title('count of passenger Based on  '+ t1)

    sns.countplot(t1,hue=t2,data=df,ax=ax[1])

    ax[1].set_title(t1 + ': Survived vs dead')

    a=plt.show()

    return a
titanic.train.columns
plot_re(train_df,'Sex','Survived')
group_by(train_df,'Sex','Survived')
train_df['Age_bin'] = pd.cut(train_df['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])

train_df['Age_bin'].head()
group_by(train_df,'Age_bin','Survived')
plot_re(train_df,'Age_bin','Survived')
facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_df['Age'].max()))

facet.add_legend()


train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Title'].value_counts()
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')

train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')

train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
train_df['Title'].value_counts()
plot_re(train_df,'Title','Survived')
def or_plot(df,t1='',t2=''):

    f,ax=plt.subplots(1,2,figsize=(10,6))

    df[t1].value_counts().plot.bar(ax=ax[0],color='Green')

    ax[0].set_title('Number Of Passenger By '+t1)

    ax[0].set_xlabel("Score of :"+t1)

    ax[0].set_ylabel('Count')

    sns.countplot(t1,hue=t2,data=train_df,ax=ax[1],palette="spring")

    ax[1].set_title(t1+':Survived vs Dead')

    a=plt.show()

    return a
or_plot(train_df,'Pclass','Survived')
or_plot(train_df,'Embarked','Survived')
or_plot(train_df,'Parch','Survived')
or_plot(train_df,'SibSp','Survived')
# Create new feature FamilySize as a combination of SibSp and Parch

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['FamilySize'].value_counts()
or_plot(train_df,'FamilySize','Survived')
# Create new feature IsAlone from FamilySize

train_df['Alone'] = 0

train_df.loc[train_df['FamilySize'] == 1, 'Alone'] = 1
train_df.Alone.value_counts()
or_plot(train_df,'Alone','Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(10,10))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':16}

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(train_df)
# most correlated features

corrmat = train_df.corr()

top_corr_features = corrmat.index[abs(corrmat["Survived"])>=0.05]

plt.figure(figsize=(10,10))

g = sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="Oranges")
g = sns.pairplot(train_df[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])