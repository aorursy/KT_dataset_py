! pip install plotly==3.7.1
import pandas as pd

import numpy as np

import seaborn as sns

sns.set()



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)
# Path of datasets

train_df_raw = pd.read_csv('../input/train.csv')

train_df_raw.head()
def preprocess_data(df):

    

    processed_df = df.copy()

    processed_df['Embarked'].fillna('C', inplace=True)

    processed_df['Cabin'].fillna('U', inplace=True)

    processed_df['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in processed_df['Name']), index=processed_df.index)

    processed_df['Title'] = processed_df['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    processed_df['Title'] = processed_df['Title'].replace(['Mlle', 'Ms'], 'Miss')

    processed_df['Title'] = processed_df['Title'].replace('Mme', 'Mrs')

    processed_df['FamillySize'] = processed_df['SibSp'] + processed_df['Parch'] + 1

    processed_df['IsAlone'] = np.where(processed_df['FamillySize']!=1, 0, 1)

    processed_df['IsChild'] = processed_df['Age'] < 18

    processed_df['Deck'] = processed_df['Cabin'].str[:1]

    processed_df['FamillySize'][processed_df['FamillySize'].between(1, 5, inclusive=False)] = 2

    processed_df['FamillySize'][processed_df['FamillySize']>5] = 3

    processed_df['FamillySize'] = processed_df['FamillySize'].map({1: 'Alone', 2: 'Medium Familly', 3: 'Large Familly'})

    processed_df['IsAlone'] = processed_df['IsAlone'].map({0: 'Not Alone', 1: 'Alone'})

    processed_df['Embarked'] = processed_df['Embarked'].map({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})

    processed_df['Pclass'] = processed_df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'})

    processed_df = processed_df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], 1)  



    return processed_df
train_df = preprocess_data(train_df_raw)

train_df.head()
train_df['Sex'].iplot(kind='hist', 

                      yTitle='Count', 

                      title='Sex Distribution', 

                      colors='#A0DDFF')
train_df[['Embarked', 'Pclass', 'FamillySize' ,'Title']].iplot(kind='hist',

                                                               yTitle='Count', 

                                                               title='Variable Distribution', 

                                                               subplots=True, 

                                                               shape=(2, 2))
train_df_fareoutliers_as_nan = train_df.copy()

train_df_fareoutliers_as_nan[train_df_fareoutliers_as_nan['Fare'] > 100] = np.nan

train_df_fareoutliers_as_nan[['Age', 'Fare']].iplot(kind='hist',

                                                    yTitle='Count', 

                                                    title='Age & Fare Distribution', 

                                                    subplots=True, 

                                                    shape=(2, 1),

                                                    colors=['#7D7ABC', '#00CECB'])
train_df_without_fareoutliers = train_df[train_df['Fare'] < 100]

train_df_without_fareoutliers.pivot(columns='Pclass', values='Fare').iplot(kind='box',

                                                                           title='Age Distribution by Pclass',

                                                                           colors=['#002277', '#51344D', '#F58A07'])
train_df.pivot(columns='Pclass', values='Age').iplot(

        kind='hist',

        barmode='stack',

        yTitle='Count',

        xTitle='Ages',

        title='Age Distribution by Pclass',

        colors=['#5AC7E6', '#FFB045', '#6A6A6A'])
f,axes = plt.subplots(1,2,figsize=(18,8))

train_df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=axes[0],shadow=True)

sns.countplot('Survived',data=train_df,ax=axes[1])

plt.show()
figure, axes = plt.subplots(1,2,figsize=(20,6))

sns.countplot('Sex',hue='Survived',data=train_df, palette=("Set2"), ax=axes[0])

sns.countplot('Sex',hue='Pclass',data=train_df, palette=("Set2"), ax=axes[1])

plt.show()
figure, axes = plt.subplots(1,2,figsize=(20,6))

g1 = sns.catplot(x="Embarked", y="Survived", hue="Sex", data=train_df, kind="bar", palette=("nipy_spectral"), ax=axes[0])

g2 = sns.catplot(x="Title", y="Survived", hue="Sex", data=train_df, kind="bar", palette=("nipy_spectral"), ax=axes[1])

plt.close(g1.fig)

plt.close(g2.fig)

plt.show()
figure, axes = plt.subplots(1,3,figsize=(20, 6))

g1 = sns.catplot(x="IsAlone", y="Survived", hue="Sex", data=train_df, kind="bar", palette=('Set1'), ax=axes[0])

g2 = sns.catplot(x="FamillySize", y="Survived", hue="Sex", data=train_df, kind="bar", palette=('Set1'), ax=axes[1])

g3 = sns.catplot(x="IsChild", y="Survived", hue="Sex", data=train_df, kind="bar", palette=('Set1'), ax=axes[2])

plt.close(g1.fig)

plt.close(g2.fig)

plt.close(g3.fig)

plt.show()
plt.figure(figsize=(20, 6))

train_df_without_u_deck = train_df.drop(train_df[train_df.Deck == 'U'].index)

plot = sns.countplot(x="Deck", hue="Survived", data=train_df_without_u_deck, palette=('Accent'))

plt.show()
figure, axes = plt.subplots(1,2,figsize=(20,6))

g1 = sns.boxplot(x="Deck", y="Fare", data=train_df_without_u_deck, ax=axes[0])

g2 = sns.countplot(x="Deck", hue="Pclass", data=train_df_without_u_deck, ax=axes[1])

plt.show()