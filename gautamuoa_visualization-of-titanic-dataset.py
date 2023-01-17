# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math #library for mathematical calculation
import numpy as np #library for scientific computing
import pandas as pd #library for easy-to-use data structures and data analysis tools 
import matplotlib.pyplot as plt #library for plotting
import seaborn as sns #library for plotting

#import data
df = pd.read_csv('../input/train.csv')

#print details of the dataset
print('_'*50)
print('*'*50)
print(df.info(memory_usage=False))
print('_'*50)
#drop passengerID, Name, Cabin columns
df = df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

#drop rows which has nan value
df = df.dropna(axis=0, how='any')

#map 0 to not survived and 1 to survived in survived column
df['Survived'] = df['Survived'].map({0: 'Not Survived', 1: 'Survived'})

#map 1 to First Class and 2 to Second Class and 3 to Third Class in Plcass column
df['Pclass'] = df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3:'Third Class'})

#map C to Cherbourg and Q to Queenstown and S to Southampton in Embarked column
df['Embarked'] = df['Embarked'].map({'C': 'Cherbourg', 'Q': 'Queenstown', 'S':'Southampton'})

#group passengers into bins of children, teenager, adult, senior citizen...
df['Age'] = pd.cut(df['Age'], bins=[-1, 5, 19, 60, 150], labels=['Children','Teenager','Adult','Senior Citizen'],right=True)

#set the plot
sns.set(context='notebook',style='whitegrid',font_scale=1.5)

#the two methods below is for displaying the count value on top of the bar graph
def assignAxis(df,g):
    # Get current axis on current figure   
    for i in range(0,g.axes.size):
        ax = g.fig.get_axes()[i]
        displayCount(df,ax)

def displayCount(df,ax):
    # ylim max value to be set
    y_max = df.value_counts().max() + 75    
    ax.set_ylim(top=y_max)
    
    # Iterate through the list of axes' patches
    for p in ax.patches:
        #checks if there index has 0 count        
        if(math.isnan(p.get_height())):
            ax.text(p.get_x() + p.get_width()/2, 0, 0, 
                fontsize=12, color='red', ha='center', va='bottom')
            continue
        else:            
            ax.text(p.get_x() + p.get_width()/2, p.get_height(), int(p.get_height()), 
                fontsize=12, color='red', ha='center', va='bottom')

#number of males and females
g = sns.factorplot(x='Sex', data=df,kind='count',size=4, aspect=.8,alpha=0.7,
               palette='muted').set(xlabel='Gender',ylabel='Count',title='Number of Male and Female')

assignAxis(df['Sex'],g)

#number of males and females based on Age
g = sns.factorplot(x='Age', data=df,kind='count',size=4, aspect=1.8, hue='Sex',alpha=0.7,
               palette='muted').set(xlabel='Age',ylabel='Count',title='Gender Distribution by Age')

assignAxis(df['Age'],g)

#people in different class
g = sns.factorplot(x='Pclass', data=df,kind='count',size=4, aspect=1.4,hue="Sex",alpha=0.7,
               palette='muted',order=['First Class','Second Class','Third Class']).set(xlabel='Class',ylabel='Count',title='Gender Distribution in different Classes')

assignAxis(df['Pclass'],g)

#male and female in survived different class
g = sns.factorplot(x='Pclass', data=df,kind='count',size=4, aspect=1.4,hue="Survived",alpha=0.7,
               palette='muted',order=['First Class','Second Class','Third Class']).set(xlabel='Class',ylabel='Count',title='Survival Distribution in different Classes')

assignAxis(df['Pclass'],g)            
#People Survived based on gender and age
g = sns.factorplot(x='Survived', data=df,kind='count',size=4, aspect=1.2,hue='Age',col='Sex',alpha=0.7,
               palette='muted',order=['Survived','Not Survived'],legend_out=True).set(xlabel='Survival',ylabel='Count')
g.fig.subplots_adjust(wspace=.3)

assignAxis(df['Survived'],g)


g = sns.factorplot(x='Embarked', data=df,kind='count',size=4, aspect=1.4,hue="Sex",alpha=0.7,
               palette='muted',).set(xlabel='Boarding Point',ylabel='Count',title='Ship Boarding Point')

assignAxis(df['Embarked'],g)            


g = sns.factorplot(x='Embarked', data=df,kind='count',size=4, aspect=1.4,hue="Survived",alpha=0.7,
               palette='muted',).set(xlabel='Boarding Point',ylabel='Count',title='Survival Rate from each Boarding Point')

assignAxis(df['Embarked'],g)            