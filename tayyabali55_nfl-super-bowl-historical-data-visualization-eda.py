# for basic operations

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for visualizations

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# for defining path

import os

# read the dataset

nfl_data = pd.read_csv("/kaggle/input/superbowl-history-1967-2020/superbowl.csv",index_col="Date")



# let's check the shape of the dataset



nfl_data.shape # dimensions of the data
# first look of the data

nfl_data.head(10)
# Statistical description of NFL points

round(nfl_data.describe())
sns.set(font_scale=1.4)
nfl_data['Win_vs_lost'] = nfl_data['Winner'] +"  VS  " + nfl_data['Loser'] 

plt.figure(figsize=(20,20))

plt.xticks( rotation=90)



sns.countplot(y="Win_vs_lost",data=nfl_data,orient='h',order=nfl_data['Win_vs_lost'].value_counts().sort_values(ascending=False).index)

sns.set_style("whitegrid")





plt.title("Faced Eaachother") 

plt.xlabel("# of Time Teams Faced Eachother") 

plt.ylabel("NFL Teams") 

plt.show()


plt.figure(figsize=(20,20))

plt.xticks( rotation=75)





sns.scatterplot(x=nfl_data.index,y=nfl_data['Win_vs_lost'],hue=nfl_data['Winner'],s=150)

sns.set_style("whitegrid")





plt.title("Number of Time Faced and Winers By Year") 

plt.ylabel("NFL Teams") 

plt.xlabel("Game Years") 



plt.figure(figsize=(20,20))

plt.xticks( rotation=90)



sns.scatterplot(x=nfl_data['Winner'],y=nfl_data['Loser'],hue=nfl_data['Winner'],s=200)

sns.set_style("whitegrid")





plt.title("Winners vs Losers Teams") 

plt.ylabel("Losers") 

plt.xlabel("Winners") 

plt.figure(figsize=(20,10))

plt.xticks( rotation=75)



sns.countplot(x="Winner",data=nfl_data,order=nfl_data['Winner'].value_counts().sort_values(ascending=False).index)

sns.set_style("whitegrid")





plt.title("Wins by each Team") 

plt.ylabel("# of Wins") 

plt.xlabel("NFL Teams") 

plt.show()



plt.figure(figsize=(20,20))

plt.xticks( rotation=75)





sns.scatterplot(x=nfl_data.index,y=nfl_data['Winner'],hue=nfl_data['Winner'],s=150)

sns.set_style("whitegrid")





plt.title("Wins of each Team By Year") 

plt.ylabel("NFL Teams") 

plt.xlabel("Game Years") 



plt.figure(figsize=(20,10))

plt.xticks( rotation=75)



sns.countplot(x="Loser",data=nfl_data,order=nfl_data['Loser'].value_counts().sort_values(ascending=False).index)

sns.set_style("white")





plt.title("Loses by each Team") 

plt.ylabel("# of Loses") 

plt.xlabel("NFL Teams") 



plt.figure(figsize=(20,20))

plt.xticks( rotation=75)



sns.scatterplot(x=nfl_data.index,y=nfl_data['Loser'],hue=nfl_data['Loser'],s=150)

sns.set_style("whitegrid")





plt.title("Loss of each Team By Year") 

plt.ylabel("NFL Teams") 

plt.xlabel("Game Years") 



plt.figure(figsize=(20,10))

plt.xticks( rotation=75)



sns.countplot(x="SB",data=nfl_data,order=nfl_data['SB'].value_counts().sort_values(ascending=False).index)

sns.set_style("white")





plt.title("Super Bowls Played Counts") 

plt.ylabel("# of SB's") 

plt.xlabel("Super Bowls") 



plt.figure(figsize=(20,10))

plt.xticks( rotation=75)



sns.barplot(x=nfl_data['Winner'],y=nfl_data['Winner Pts'])

sns.set_style("whitegrid")





plt.title("Wins NFL Teams by Points") 

plt.ylabel("Points") 

plt.xlabel("NFL Teams") 



plt.figure(figsize=(20,10))

plt.xticks( rotation=75)



sns.barplot(x=nfl_data['Loser'],y=nfl_data['Loser Pts'])

sns.set_style("whitegrid")





plt.title("Loses NFL Teams by Points") 

plt.ylabel("Points") 

plt.xlabel("NFL Teams") 



plt.figure(figsize=(20,10))

plt.xticks( rotation=75)

sns.set_style("white")



sns.distplot(a=nfl_data['Winner Pts'], label="Winners")

sns.distplot(a=nfl_data['Loser Pts'], label="Loser")



plt.title("Histogram of Petal Lengths, by Species")



# Force legend to appear

plt.legend()


plt.figure(figsize=(20,10))

plt.xticks( rotation=75)



sns.countplot(x="MVP",data=nfl_data,order=nfl_data['MVP'].value_counts().sort_values(ascending=False).index,palette="rocket")

sns.set_style("whitegrid")





plt.title("MVP of Game Nominated") 

plt.ylabel("# of Nominations") 

plt.xlabel("NFL Players") 



plt.figure(figsize=(20,20))

plt.xticks( rotation=75)



sns.scatterplot(x=nfl_data.index,y=nfl_data['MVP'],s=150)

sns.set_style("whitegrid")





plt.title("MVP of Game Nominated by Year") 

plt.ylabel("Players") 

plt.xlabel("Game Years") 

plt.figure(figsize=(20,10))

plt.xticks( rotation=85)



sns.countplot(x="Stadium",data=nfl_data,order=nfl_data['Stadium'].value_counts().sort_values(ascending=False).index,palette="rocket")



sns.set_style("whitegrid")





plt.title("Game played in most stadiums") 

plt.ylabel("# of Time Playes in Stadiums") 

plt.xlabel("Stadiums") 

plt.show()



plt.figure(figsize=(20,20))

plt.xticks( rotation=75)



sns.scatterplot(x=nfl_data.index,y=nfl_data['Stadium'],hue=nfl_data['City'],s=200)

sns.set_style("whitegrid")







plt.title("Game Hosted by each Stadium and City") 

plt.ylabel("Stadiums") 

plt.xlabel("Year") 

plt.figure(figsize=(20,10))

plt.xticks( rotation=90)



sns.countplot(x="City",data=nfl_data,order=nfl_data['City'].value_counts().sort_values(ascending=False).index,palette="rocket")



sns.set_style("whitegrid")





plt.title("Game played in most cities") 

plt.ylabel("# of Time Playes in City") 

plt.xlabel("US Cities") 

plt.show()



plt.figure(figsize=(20,20))

plt.xticks( rotation=75)



sns.scatterplot(x=nfl_data.index,y=nfl_data['City'],hue=nfl_data['State'],s=150)

sns.set_style("whitegrid")







plt.title("Game in the city in states by year") 

plt.ylabel("Cities") 

plt.xlabel("Year") 
