# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# IMPORTING THE GOOD STUFF
import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

train_dataframe=pd.read_csv("../input/train.csv")
test_dataframe=pd.read_csv("../input/test.csv")
print("Got them!")
# Visualisation by pivoting
print(train_dataframe[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# it puts .sort_values(by='Survived',ascending=False) at the end, but that is not necessary here
# Visualising by pivoting - trying to look at sex and survived
sex_df=train_dataframe[['Sex','Survived']]
print(sex_df.groupby(['Sex'], as_index=False).head())
print("_"*40)
print(sex_df.groupby(['Sex'], as_index=False).mean())
print("_"*40)
print(sex_df.groupby(['Sex'],as_index=False).max())
print("_"*40)
print(sex_df.groupby(['Sex']).max())
print("_"*40)
print(sex_df.groupby(['Sex']).std())

agecut=pd.cut(train_dataframe['Age'], 6)
agecut
survived=train_dataframe[train_dataframe['Survived']==1]
died=train_dataframe[train_dataframe['Survived']==0]
# Visualisation - plots - Age
# what is on the y axis? Frequency density?
g = sns.FacetGrid(train_dataframe, col='Embarked')
g.map(sns.boxplot, 'Survived', 'Age')

ports=train_dataframe['Embarked'].unique()
surv1 = survived[survived['Embarked']==ports[0]][['Age']]
surv2 = survived[survived['Embarked']==ports[1]]
surv3 = survived[survived['Embarked']==ports[2]]
die1 = died[died['Embarked']==ports[0]]['Age']
die2 = died[died['Embarked']==ports[1]]
die3 = died[died['Embarked']==ports[2]]

h = plt.figure(figsize=(100,5))
hax1=h.add_axes([0,0,.3, 1])
hax1.set_xlim(0,100)
plt.hist(surv1[['Age']], bins=40, color='green', alpha=.3)
plt.hist(die1, bins=40, color='red', alpha=.3)
print(surv1.info())
hax1.set_title(str(ports[0]))
hax2=h.add_axes([.35, 0, .3, 1])
hax2.set_xlim(0,100)
plt.hist(surv2[['Age']], bins=40, color='green', alpha=.3)
plt.hist(die2[['Age']], bins=40, color='red', alpha=.3)
hax2.set_title(str(ports[1]))
hax3=h.add_axes([.7,0,.3,1])
hax3.set_xlim(0,100)
plt.hist(surv3[['Age']], bins=40, color='green', alpha=.3)
plt.hist(die3[['Age']], bins=40, color='red', alpha=.3)
hax3.set_title(str(ports[2]))

#then for each one, plot a histogram of died and survived by age
#make the limits the same for all axes



sh = sns.FacetGrid(train_dataframe, col='Embarked')
sh.map(plt.hist, 'Age', bins=20)

s = sns.FacetGrid(train_dataframe,col='Survived')
s.map(plt.hist,'Age',bins=30)

# what about plotting Age against chance of survival? THIS DOESN'T WORK
# age_df=train_dataframe[['Age','Survived']].groupby(['Age']).mean()
# t = sns.FacetGrid(age_df,col='Survived')
# t.map(plt.hist,'Age',bins=80)
# My pivots to try and see:
# did groups survive and die together?
# this is not very helpful. Really I want # people on ticket/in cabin
# but when the ticket is split across 2 or more cabins, they do seem to mostly die or mostly survive
#is this truncating the dataset to remove those who didn't have cabins?
cabin_df=train_dataframe[['Cabin','Age','Ticket','Survived']]
cabin_df.groupby(['Ticket','Cabin'],as_index=False).mean()
# Making an array of histograms to compare class and age
# size is the size but if it's very big then the labels just get very small
# aspect is the scale factor from the vertical to horizontal axis (regardless of units etc, just aspect
# ratio of the histogram)
# not filling them in provides a reasonable graph which is square
age_Pclass_grid = sns.FacetGrid(train_dataframe, col='Survived', row='Pclass', size=2.2, aspect = 1.6)
age_Pclass_big = sns.FacetGrid(train_dataframe, col='Survived', row='Pclass')
age_Pclass_skinny=sns.FacetGrid(train_dataframe,col='Survived',row='Pclass', size=2.2, aspect = 9)
A = age_Pclass_grid.map(plt.hist, 'Age', alpha=.5, bins=20)
B = age_Pclass_grid.map(plt.hist, 'Age', alpha=.1, bins=20)
# creating 3 in this way doesn't do anything. You only get one. I think you have to give the grids 
# different names to make different plots
C = age_Pclass_grid.map(plt.hist, 'Age', alpha=.5, bins=20)
age_Pclass_grid.add_legend();
# alpha is the strength of the colour of the bars 0 - 1; default is 1 I think
# bins is the number of classes, I think default is 10
age_Pclass_big.map(plt.hist, 'Age', alpha=1, bins=20)
age_Pclass_skinny.map(plt.hist, 'Age')
# More visualisations also part port of embark
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_dataframe, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# Now I can do what I wanted to do before with the age, since this gives probability of survival
# on the y axis
# So I wanted age and survived only, maybe I can split that by the 3 classes
age_survive= sns.FacetGrid(train_dataframe,aspect=5)
age_survive.map(sns.pointplot,'Age','Survived','Pclass', palette='deep')
age_survive.add_legend()
# bit of a mess because age is not scaled uniformly along the axis - the spaces between age < 1 are too big
age_histogram = sns.FacetGrid(train_dataframe,aspect=5)
# it doesn't work; it only takes one argument, 'Survived' it thinks it the number of bins.
# age_histogram.map(plt.hist,'Age','Survived', bins=40)
# Trying to use histogram for grouped data
# I want to make my histogram by 'Age' with probability of survival on the y axis
# I would like to be able to split it by 'Sex' or 'Class'
# instead of splitting it into different histograms by whether or not they survived
# But it doesn't work at all; what's supposed to go on the y axis? Not count/frequency any more
age_sex_class=train_dataframe.groupby(['Pclass','Sex','Age']).mean()
asc_grid = sns.FacetGrid(age_sex_class, aspect=3)
asc_grid.map(plt.hist,'Age')
# experimenting with jointplot
sns.jointplot(x='Age',y='Survived',data=train_dataframe)
sns.jointplot(x='Age',y='Survived',kind='hex',data=train_dataframe)
sns.jointplot(x='Age',y='Pclass',data=train_dataframe)
# still doesn't do what I want to do - showing frequency not probabilities, so in either case
# people in their 20s have highest survival density because there are most of them

# experimenting with barplot, pointplot for subsets of data frame
# another attempt using the line idea
child_survive= sns.FacetGrid(train_dataframe[train_dataframe['Age']<18],aspect=5)
child_survive.map(sns.barplot,'Age','Survived','Sex', palette='deep',ci=False)
# height of bar showns mean and line is 95% confidence interval
child_survive.add_legend()
# can't restrict on both sides - think I need to redefine whole data frame
#young_adult_survive= sns.FacetGrid(train_dataframe[35>train_dataframe['Age'] and train_dataframe['Age']>=18],aspect=5)
#young_adult_survive.map(sns.barplot,'Age','Survived','Sex')
#mid_adult_survive= sns.FacetGrid(train_dataframe[50>train_dataframe['Age']>=35],aspect=5)
#mid_adult_survive.map(sns.barplot,'Age','Survived','Sex')
#old_adult_survive= sns.FacetGrid(train_dataframe[65>train_dataframe['Age']>=50],aspect=5)
#old_adult_survive.map(sns.barplot,'Age','Survived','Sex')

# want to see the people who died on the graph but that didn't work
oldest_adult_survive=sns.barplot('Age','Survived',data=train_dataframe[train_dataframe['Age']>=60])
oldest_adult_survive.set_ylim(-1,1)
# experimenting with barplot, pointplot for subsets of data frame
sns.pointplot('Age','Survived','Sex',data=train_dataframe[train_dataframe['Age']>=60],ci=False)
# experimenting with barplot, pointplot for subsets of data frame
sns.pointplot('Age','Survived',data=train_dataframe[train_dataframe['Age']>=60],ci=False)
# from tutorial - categorical plots
# I don't think this is very useful
# I think this is better summed up in the pivots below
grid = sns.FacetGrid(train_dataframe, row='Embarked', col='Survived', aspect=1.6)
# something strange is happening here; this isn't right
grid.map(sns.countplot, 'Sex', alpha=.5)
grid.add_legend()
grid2 = sns.FacetGrid(train_dataframe, row='Embarked', col='Survived', aspect=1.6)
grid2.map(sns.barplot, 'Fare', 'Sex', alpha=.5, ci=None)
grid2.add_legend()
grid3 = sns.FacetGrid(train_dataframe, row='Embarked', col='Survived', aspect=1.6)
grid3.map(sns.barplot,'Fare', alpha=.5, ci=None)
grid3.add_legend()
# Checking whether I managed to show what I wanted to
train_dataframe[train_dataframe['Embarked']=='S'][train_dataframe['Sex']=='female'].describe()
# Pivoting to show different information
print(train_dataframe[['Survived','Embarked']].groupby(['Embarked']).mean())
print('_'*50)
print(train_dataframe[['Fare','Survived','Embarked']].groupby(['Embarked','Survived']).mean())
print('_'*50)
print(train_dataframe[['Fare','Survived','Embarked','Sex']].groupby(['Embarked','Sex','Survived']).mean())
print('_'*50)
# does this do what I think it does?
print(train_dataframe[['Fare','Survived','Embarked','Sex']].groupby(['Embarked','Sex','Survived']).count())
# IT DOES! SO NOW I CAN DO WHAT I WANTED TO DO FOR CABIN AND TICKET
# checking whether iit is showing what I wanted
train_dataframe[train_dataframe['Embarked']=='S'][train_dataframe['Sex']=='male'][train_dataframe['Survived']==1].describe()
# more visualisations pivoting by ticket, cabin
train_dataframe[['Ticket','Survived','Cabin']].groupby(['Cabin','Survived']).count().sort_values(by='Ticket')
# more visualisations pivoting by ticket, cabin
train_dataframe[['Ticket','Survived','Cabin','Sex']].groupby(['Ticket','Survived']).count().sort_values(by='Sex')
# really I want to sort by count(Ticket) only
# but this doesn't work:
# train_dataframe[['Ticket','Survived','Cabin','Sex']].groupby(['Ticket','Survived']).count().sort_values(by=['Ticket'].count('Sex'))
# checking what the above shows
train_dataframe[train_dataframe['Ticket']=='PC 17755']
# I saw above that there were 3 'Sex' but only 2 'Cabin'; 
# This is true - Miss. Anna Ward's cabin is not listed
# experimenting with new plot types
sns.violinplot(x='Pclass',y='Age',hue='Survived',data=train_dataframe, split=True)
# I was hoping this would show the relative sizes of the datasets, but actually 
# it doesn't appear to offer anything different to the plots below
# experimenting with new plot types
sns.violinplot(x='Pclass',y='Age',hue='Survived',data=train_dataframe)
# experimenting with new plot types
sns.boxplot('Survived','Age','Pclass',train_dataframe)
sns.violinplot('Survived','Age','Pclass',train_dataframe)
# experimenting with new plot types
sns.violinplot('Survived','Age','Sex',train_dataframe)
# how many people in the test data were on the same tickets as those in the training data?
t=0
for i in train_dataframe['Ticket']:
    for j in test_dataframe['Ticket']:
        if i==j:
            t+=1
print(t)
c=0
for i in train_dataframe['Cabin']:
    if i=='NaN':
        break
    for j in test_dataframe['Ticket']:
        if i==j:
            c+=1
print(c)
# So 297 people in the test data shared tickets with people in the training data - why so many?
# And why did noone share a cabin with anyone in the training data?
