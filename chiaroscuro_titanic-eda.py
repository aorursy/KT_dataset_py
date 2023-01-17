#Import modules

import pandas as pd

from pandas import Series,DataFrame

# Set up the Titanic csv file as a DataFrame

titanic_df = pd.read_csv('../input/train.csv')



# Module for analysis and visualization

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#preview of the data

titanic_df.head()
#Check gender

sns.countplot('Sex',data=titanic_df)
# Further breakdown by seperate by class

sns.countplot('Pclass',data=titanic_df,hue='Sex')
# For child

def male_female_child(passenger):

    # Take the Age and Sex

    age,sex = passenger

    # Compare the age, otherwise leave the sex

    if age < 16:

        return 'child'

    else:

        return sex

    

# A new column called 'person'

titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
sns.countplot('Pclass',data=titanic_df,hue='person')
# Histogram for Age

titanic_df['Age'].hist(bins=70)
# A quick overall comparison of male,female,child

titanic_df['person'].value_counts()
# use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(titanic_df, hue="person",aspect=4)

fig.map(sns.kdeplot,'Age',shade= True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
# Check for class use the same skill

fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)

fig.map(sns.kdeplot,'Age',shade= True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
# First drop the NaN values in Cabin column and create a new object, deck

deck = titanic_df['Cabin'].dropna()
# Quick preview of the decks

deck.head()


# Loop to grab first letter

levels = []

for level in deck:

    levels.append(level[0])    

   

cabin_df.columns = ['Cabin']

cabin_df = DataFrame(levels) 

cabin_df.columns = ['Cabin']

cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.countplot('Cabin',data=cabin_df,palette='winter_d')
# A quick overall comparison of male,female,child

titanic_df['Embarked'].value_counts()

sns.countplot('Embarked',data=titanic_df,hue='Pclass',order=['C','Q','S'])
# adding a new column to define alone

titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp

# Look for >0 or ==0 to set alone status

titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

sns.countplot('Alone',data=titanic_df,palette='Blues')
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

sns.countplot('Survivor',data=titanic_df,palette='Set1')
#Class survival rates

sns.factorplot('Pclass','Survived',data=titanic_df)
# Includes gender this time

sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)
# A linear plot on age versus survival using hue for class seperation

generations=[10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)
#Gender

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)