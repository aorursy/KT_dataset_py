#importing necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly as py
#now we will load the FIFA 19 dataset

#we will use pandas library; since the dataset is in csv format

#we will use the syntax to make it into tabular form

data=pd.read_csv("../input/data.csv")
#lets see the summary of the dataset

data.describe()
#first five rows of the dataset

data.head(5)
#last five rows

data.tail(5)
#lets see how many rows and columns we have in our dataset

print("Number of (rows,columns):",data.shape)
#checking if there is any NULL value in the dataset

data.isna().sum()
#we saaw that most of the data in 'Loaned From' column is not assigned, hence we will drop it

data.drop('Loaned From',axis=1,inplace=True)
#now the data which have NA values, we will fill them with the mean value of that column

data.fillna(data.mean(),inplace=True)
#we will check again if after assigning the mean value to the cells of the originally NA values; if there is any cell which has NA value

data.isna().sum()
#there are still cells in which the mean value could not be assigned. This may be because those columns have strings. So we will assign a value "Unassigned" to the dataset

data.fillna("Unassigned",inplace=True)
#after assigning the term, we shall check again whether we have attained a clean data set or not

data.isna().sum()
#as we started our analysis with the summary of the dataset. We will make a heatmap for the same.

plt.figure(figsize=(40,40))

p=sns.heatmap(data.corr(),annot=True,cmap='YlOrBr')

p.set_title(label='Heatmap',fontsize=25)

p
# Lets see the top 15 country-wise distribution of players

fif_countries = data['Nationality'].value_counts().head(15).index.values

fif_countries_data = data.loc[data['Nationality'].isin(fif_countries),:]
#we will make a simple visualization for the 15 countries data

#We will make a basic Bar Plot

sns.set(style="dark")

plt.figure(figsize=(25,10))

p=sns.barplot(x='Nationality',y='Overall',data=fif_countries_data)

p.set(xlabel='Country', ylabel='Total')
#Box Plot

sns.set(style="ticks")

plt.figure(figsize=(25,10))

p=sns.boxplot(x='Nationality',y='Overall',data=fif_countries_data)

p.set(xlabel='Country', ylabel='Total')
ten_countries = data['Nationality'].value_counts().head(10).index.values

ten_countries_data = data.loc[data['Nationality'].isin(ten_countries),:]

sns.set(style="ticks")

plt.figure(figsize=(15,10))

p=sns.boxplot(x='Nationality',y='Potential',data=ten_countries_data)
#i will make a dataframe of the dataset. but before that I will choose only those columns which can be used for analysis

chosen_columns = ['Name','Age','Nationality','Overall','Potential','Special',

                  'Acceleration','Aggression','Agility','Balance','BallControl',

                  'Body Type','Composure','Crossing','Curve','Club','Position',

                  'Dribbling','FKAccuracy','Finishing','GKDiving','GKHandling',

                  'GKKicking','GKPositioning','GKReflexes','HeadingAccuracy',

                  'Interceptions','International Reputation','Jersey Number',

                  'Jumping','Joined','LongPassing','LongShots','Marking',

                  'Penalties','Positioning','Preferred Foot','Reactions',

                  'ShortPassing','ShotPower','Skill Moves','SlidingTackle',

                  'SprintSpeed','Stamina','StandingTackle','Strength','Value',

                  'Vision','Volleys','Wage','Weak Foot','Work Rate']
df=pd.DataFrame(data,columns=chosen_columns)
df
#The number of footballers for each position.

tot_players=[df["Position"].value_counts()]

tot_players
features = ('Acceleration', 'Aggression', 'Agility', 'Balance',

    'BallControl', 'Composure', 'Crossing', 'Dribbling', 'FKAccuracy',

    'Finishing', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning',

    'GKReflexes', 'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking')

#grouping the players by features and by positions

for i, val in df.groupby(df['Position'])[features].mean().iterrows():

    print('Position {}: {}, {}, {}, {}, {}'.format(i, *tuple(val.nlargest(5).index)))
#histogram for the age

plt.figure(1,figsize=(15,5))

df['Age'].plot(kind='hist',bins=50)

plt.show()
#scatter plot of Age vs Overall

plt.figure(1,figsize=(10,5))

sns.regplot(df['Age'],df['Overall'])

plt.title('Age vs Overall Scatter Plot')

plt.show()
#scatter plot of Age vs Potential

plt.figure(1,figsize=(12,8))

sns.regplot(df['Age'],df['Potential'])

plt.title('Age vs Potential Scatter Plot')

plt.show()
#Choosing specific columns from the data frame to make regression plots

abilities=['Reactions','ShotPower','Jumping','SprintSpeed',

'Stamina','Agility','Strength','Vision']

import random

class fast_plot():

    def __init__(self):

        c = ['r' , 'g' , 'b' , 'y' , 'orange' , 'grey' , 'lightcoral' , 'crimson' , 

            'springgreen' , 'teal' , 'c' , 'm' , 'gold' , 'skyblue' , 'darkolivegreen',

            'tomato']

        self.color = c

    #have to make many plots at the same place

    def regplot_one_vs_many(self , x  , y  , data , rows , cols):

        color_used = []

        n = 0

        for feature in y:

            for i in range(1000):

                colour = random.choice(self.color)

                if colour not in color_used:

                    color_used.append(colour)

                    break

            n+=1 

            plt.subplot(rows , cols , n)

            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

            sns.regplot(x  = x , y = feature , data = data ,  color = colour)

plots=(fast_plot())

#for making regression plots defined using function

plt.figure(1,figsize=(12,8))

plots.regplot_one_vs_many(x='Age',y=abilities,data=df,rows=2,cols=4)

plt.show()
#count of players Position Wise

plt.figure(1,figsize=(15,8))

p = sns.countplot(x = 'Position', data = df,palette='inferno_r')

p.set_title(label='Count of Players', fontsize=15)
top15players=df.sort_values(by='Overall',ascending=False).head(15)

plt.figure(figsize=(25,10))

p=sns.barplot(x='Name',y='Overall',data=top15players)

p = plt.ylim(85,95)

p=plt.xticks(rotation='vertical')
#Stamina vs Sprint Speed Plot

data.plot(kind = 'scatter' , x='Stamina' , y = 'SprintSpeed' , alpha = .5 , color = 'b')

plt.xlabel('Stamina')

plt.ylabel('Sprint Speed')

plt.title('Stamina-Sprint Speed Scatter Plot')

plt.show()
plt.figure(1,figsize=(10,5))

sns.regplot(df['Age'],df['SprintSpeed'])

plt.title('Age vs Sprint Speed Plot')

plt.show()
p=data.groupby('Overall')['Potential'].mean().plot()

plt.title('Overall vs Potential')

plt.ylabel("Potential",rotation=90)

sns.set(style='darkgrid')
sns.distplot(df['Overall'])
sns.set()

cols=['Acceleration','Balance','BallControl','Crossing','Curve','Stamina','Value']

sns.pairplot(df[cols],height=2)

plt.show()
plt.figure(figsize=(5,5))

sns.countplot(df['Preferred Foot'])