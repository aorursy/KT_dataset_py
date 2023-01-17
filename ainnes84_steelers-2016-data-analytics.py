# First we must import the necessary modules for data manipulation and visual representation



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as matplot

import seaborn as sns

%matplotlib inline
# Now that we have all the modules loaded in we can now read the analytics csv

# file and store our dataset into a dataframe called "NFL"

NFL=pd.read_csv("../input/NFL by Play 2009-2016 (v2).csv",low_memory=False)

NFL.info()
NFL.isnull().any()
# We are missing some data but we will scrub what exactly we want

# So let's first take a look at what we are working with

NFL.head()
# So far this dataset contains a lot of data. Let's trim it down to

# the Steelers stats from last season which is 2016

Steelers = NFL[((NFL["HomeTeam"] == 'PIT') | (NFL["AwayTeam"] == 'PIT')) 

             & (NFL["Season"] == 2016) & (NFL['Touchdown'] == 1)]

grouped = Steelers.groupby(by='Date')

len(grouped)
# Now that we have all the Steelers data let's look at all the offensive plays

offense = Steelers[(Steelers["DefensiveTeam"] != 'PIT')]



# We are going to sort it by yards gained including the top 100 plays

Top_Plays = offense.sort_values(by='Yards.Gained',ascending=False)[:48]



# Now let's make sure we can see both rushers and recievers

Top_Plays['scorer'] = Top_Plays["Rusher"]

Top_Plays['scorer'].fillna(Top_Plays['Receiver'], inplace = True)



# Let's look at what we have

Touchdowns = Top_Plays[['PlayType',

          'down',

          'Yards.Gained',

          'Date',

          'qtr',

          'desc',

          'scorer',

          'Rusher',

          'Receiver']]

Touchdowns.head()
# Currently, each row indicates a play. Each play also represents a

# touchdown scored. As we can see we have a total of 48 touchdowns scored

# and 9 features that we can observe

Touchdowns.shape
# Let's look at our datatypes

Touchdowns.dtypes
# To better describe the data we can use the describe function

Touchdowns.describe()
# Let's also take a look at each player who scored

Player = Touchdowns.groupby('scorer')

Player.mean()
# Let's see who has the most touchdowns out of all 48

x = sns.countplot(Top_Plays['scorer'])

plt.setp(x.get_xticklabels(), rotation=45)

plt.tight_layout()

plt.show()
# Let's look at the play breakdown with a bar plot

sns.countplot(x="PlayType", data=Top_Plays)
# Next we will look at a scoring distribution from down, 

# yards gained, and quarter.

fig, axs = plt.subplots(ncols = 3, figsize=(12,6))

sns.distplot(Top_Plays["down"], ax=axs[0])

sns.distplot(Top_Plays["Yards.Gained"], ax=axs[1])

sns.distplot(Top_Plays["qtr"], ax=axs[2])

plt.tight_layout()

plt.show()
#So who scored the most between Receivers and Rushers?

# First let's look at rushers:

runs = offense[(offense["PlayType"] == 'Run')]

sns.countplot(x="Rusher",data=runs)
# Next we will look at Receivers:

passes = offense[(offense["PlayType"] == 'Pass')]

x = sns.countplot(x="Receiver",data=passes)

plt.setp(x.get_xticklabels(), rotation=45)

plt.show()