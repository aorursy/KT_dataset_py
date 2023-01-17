import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)          

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/superbowl-history-1967-2020/superbowl.csv')

df.head()
print('Total number of Super Bowls: ', len(df))
wins = pd.DataFrame(df.Winner.value_counts()).reset_index().rename(columns = {'index' : 'Team'})



# Set the width and height of the figure

plt.figure(figsize=(18,10))



# Bar chart showing average score for racing games by platform

b = sns.barplot(x=wins.Winner, y=wins.Team)



b.axes.set_title("Total number of Super Bowl wins by team",fontsize=20)

b.set_xlabel("Number of wins",fontsize=15)

b.set_ylabel("Teams",fontsize=15)

b.tick_params(labelsize=15)

lost = pd.DataFrame(df.Loser.value_counts()).reset_index().rename(columns = {'index' : 'Team'})



# Set the width and height of the figure

plt.figure(figsize=(18,10))



b = sns.barplot(x=lost.Loser, y=lost.Team)



b.axes.set_title("Total number of Super Bowl losses by team",fontsize=20)

b.set_xlabel("Number of losses",fontsize=15)

b.set_ylabel("Teams",fontsize=15)

b.tick_params(labelsize=15)

# merge the two dataframes we created earlier

appearances = wins.set_index('Team').join(lost.set_index('Team'))



# Some teams have not lost in the sb so replace NaN with zero

appearances.Loser.fillna(0, inplace=True)



# create a new column that has the total number of finals appearances

appearances['Finals Count'] = appearances.Winner + appearances.Loser



# the last two columns were float so change them to int and sort the columns

appearances.sort_values(by='Finals Count', ascending=False).astype(int)
mvp = pd.DataFrame(df.MVP.value_counts()).reset_index().rename(columns = {'index' : 'Player'})



plt.figure(figsize=(20,10))

plt.xticks(rotation=85)



b = sns.barplot(x=mvp.Player, y=mvp.MVP)

b.axes.set_title("Most MVP's by Player",fontsize=20)

b.set_xlabel("Player",fontsize=15)

b.set_ylabel("MVPS",fontsize=15)

b.tick_params(labelsize=15)

# Set figure size

plt.figure(figsize=(18,10))



# Create Scatter plot

b = sns.scatterplot(x=df.State, y=df.City, s=100)

b.axes.set_title("Locations of Super Bowl",fontsize=20)

b.set_xlabel("State",fontsize=15)

b.set_ylabel("City",fontsize=15)

b.tick_params(labelsize=15)