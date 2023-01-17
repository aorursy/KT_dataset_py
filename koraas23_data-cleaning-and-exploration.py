# Import packages and 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# Import the data

data = pd.read_csv('../input/data.csv')
# Explore the data

data.head(5)
data.describe()
# First, get rid of the useless columns and only keep the useful ones

data = data[['Name', 'Age', 'Nationality', 'Overall', 'Potential',

       'Club', 'Value', 'Wage', 'Preferred Foot',

       'Weak Foot', 'Skill Moves', 'Work Rate',

       'Body Type', 'Position', 'Height', 'Weight', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',

       'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

       'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',

       'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']]
# Now, we want only top teams from 4 best leagues in the world

# There are a couple of steps to get there.

# First, see what Clubs there are, and compile list of leagues with teams

data.Club.unique()
# We have a list of all teams, now assign the best of them to leagues

PL = ['Manchester United', 'Manchester City', 'Chelsea', 'Tottenham Hotspur', 'Liverpool', 'Arsenal']

SerieA = ['Juventus', 'Napoli', 'Milan', 'Inter', 'Lazio', 'Roma']

Bundesliga = ['FC Bayern München', 'Borussia Dortmund', 'FC Schalke 04', 'FC Schalke 04']

LaLiga = ['FC Barcelona', 'Real Madrid', 'Atlético Madrid', 'Valencia CF'] 
# Now, create a new column called 'League' and populate it with League names

# For clubs from the rest of leagues, the column is going to say 'Other'

data['League'] = np.where(data['Club'].isin(PL), "Premier League", 

                           (np.where(data['Club'].isin(SerieA), "Serie A", 

                                  (np.where(data['Club'].isin(Bundesliga), "Bundesliga", 

                                            (np.where(data['Club'].isin(LaLiga), "La Liga", "Other")))))))
# We are only going to work with the teams and leagues we selected

# Only keep the teams and leagues we want

data3 = data[(data['League'] == 'Premier League') | 

              (data['League'] == 'Serie A') |

              (data['League'] == 'Bundesliga') | 

              (data['League'] == 'La Liga')]
# Check if we have any missing values in our new dataset

sns.heatmap(data3.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
# If we want to work with 'Wage' variable, we are going to have to transorm it a little

# Here's how it looks now

data3.Wage.head()
# It's an object with special characters, number anbd letters

# Transform it into an integer

data3['Wage'] = data3['Wage'].str.replace("€", "").str.replace("K", "000").astype(int)
# We can probably do the same with the 'Value' variable

data3.Value.head()
# Data seems to be in millions, but we have to make sure all values are in the same units before we transform it

data3.Value.unique()
# As suspected some values are in thousands 

# Write a function that is going to transform those values into regular numerical values and transform and use it with our column

def value_conv(x):

    new = []

    for i in x:

        list(i)

        ending = i[-1]

        if ending is 'M':

            i = i[1:-1]

            i = float(''.join(i))

            i *= 1000000

        elif ending is 'K':

            i = i[1:-1]

            i = float(''.join(i))

            i *= 1000

        else:

            i = 0

        new.append(i)

    return new



data3['Value'] = value_conv(list(data3['Value']))
# Let's see how 'Wage' correlates to Overall Rating for different leagues

sns.scatterplot(x = 'Overall', y = 'Wage', hue = 'League', data = data3)
# We can actually check the correlation between all variables

corrmat = data3.corr()

sns.heatmap(corrmat, square = True, cmap = 'YlGnBu')
# We can check relationships between selected variables

sns.set(style = 'whitegrid')

cols = ['Age', 'Wage', 'Overall', 'Preferred Foot', 'Height', 'Skill Moves', 'Weak Foot']

sns.pairplot(data3[cols], size = 2.5)
sns.boxplot(x = 'League', y = 'Wage', data = data3, palette = 'cool')
# We can define the outliers

data3.drop(data3[data3['League'] == 'Premier League'][data3['Wage'] > 299000].index, inplace = True)

data3.drop(data3[data3['League'] == 'Serie A'][data3['Wage'] > 150000].index, inplace = True)

data3.drop(data3[data3['League'] == 'Bundesliga'][data3['Wage'] > 140000].index, inplace = True)

data3.drop(data3[data3['League'] == 'La Liga'][data3['Wage'] > 295000].index, inplace = True)
sns.boxplot(x = 'League', y = 'Wage', data = data3, palette = 'cool')