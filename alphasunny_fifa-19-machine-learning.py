# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
fifa_df = pd.read_csv("../input/data.csv")
fifa_df.head(5)
fifa_df.info()
# Try to get some useful features and this featuref is not null

useful_feat     = ['Name',

                   'Age',

                   'Photo', 

                   'Nationality', 

                   'Flag',

                   'Overall',

                   'Potential', 

                   'Club', 

                   'Club Logo', 

                   'Value',

                   'Wage',

                   'Preferred Foot',

                   'International Reputation',

                   'Weak Foot',

                   'Skill Moves',

                   'Work Rate',

                   'Body Type',

                   'Position',

                   'Joined', 

                   'Contract Valid Until',

                   'Height',

                   'Weight',

                   'Crossing', 

                   'Finishing',

                   'HeadingAccuracy',

                   'ShortPassing', 

                   'Volleys', 

                   'Dribbling',

                   'Curve',

                   'FKAccuracy',

                   'LongPassing',

                   'BallControl',

                   'Acceleration',

                    'SprintSpeed',

                   'Agility',

                   'Reactions', 

                   'Balance',

                   'ShotPower', 

                   'Jumping',

                   'Stamina', 

                   'Strength',

                   'LongShots',

                   'Aggression',

                   'Interceptions',

                   'Positioning', 

                   'Vision', 

                   'Penalties',

                   'Composure',

                   'Marking',

                   'StandingTackle', 

                   'SlidingTackle',

                   'GKDiving',

                   'GKHandling',

                   'GKKicking',

                   'GKPositioning',

                   'GKReflexes']

df = pd.DataFrame(fifa_df, columns=useful_feat)
sns.heatmap(data=df.isnull() )

# it seems there is still a lot of null data
# find the age distribution

plt.figure(1, figsize=(18, 7))

sns.countplot( x= 'Age', data=df, palette='Accent')

plt.title('Age distribution of all players')

# It seems most of the age is distributed form 19 ~ 30
# The eldest players

df.sort_values(by= 'Age', ascending=False)[['Name','Nationality', 'Club', 'Position', 'Overall', 'Age']].head(5)
# The youngest players

df.sort_values(by= 'Age')[['Name','Nationality', 'Club', 'Position', 'Overall', 'Age']].head(5)
# Age distribution in few famous clubs

vals = ['Tottenham Hotspur' , 'Juventus' , 'Paris Sain-Germain' ,'FC Bayern München',

       'Real Madrid' , 'FC Barcelona' , 'Borussia Dortmund' , 'Manchester United' , 

       'FC Porto' , 'As Monaco' , 'BSC Young Boys']

df_club_age = df.loc[df['Club'].isin(vals) & df['Age']]

plt.figure(1, figsize=(15, 7))

sns.violinplot(x = 'Club', y = 'Age', data = df_club_age )

plt.title('Age distribution in some clubs')

plt.xticks(rotation=50)

plt.show()

# Real madrid is young, Juventus is not very young, Real Madrid perform very good in recent Champions League
# Age distribution in few countries

vals = ['England' , 'Brazil' , 'Portugal' ,'Argentina',

       'Italy' , 'Spain' , 'Germany' , 'Russia' , 

       'Chile' , 'Japan' , 'India', 'France']

df_age_country = df.loc[df['Nationality'].isin(vals) & df['Age'] ]

plt.figure(1, figsize=(15, 7))

sns.violinplot(x = 'Nationality', y = 'Age', data = df_age_country)

plt.title('Age distribution in some countries')

plt.xticks(rotation = 50)

plt.show()

# It seems very average
# handle all the players

def preprocess_value(x):

    x = str(x).replace('€', '')

    if('M' in str(x)):

        x = str(x).replace('M', '')

        x = float(x) * 1000000

    elif('K' in str(x)):

        x = str(x).replace('K', '')

        x = float(x) * 1000

    return float(x)



df['Value'] = df['Value'].apply(preprocess_value)
# Value ditribution

plt.figure(1, figsize=(18, 7))

sns.countplot( x= 'Value', data=df)

plt.title('Value distribution of all players')
# find the most expensive players

df.sort_values(by='Value', ascending=False)[['Name','Nationality', 'Club', 'Position', 'Overall', 'Value']].head(5)
# Which club has the average expensive players

Club_value = df.groupby('Club')['Value'].mean()

Club_value.sort_values(ascending=False).head(5)

# These top guys is commen in Champion League
# which club has the highest total value 

club_values =df.groupby('Club')['Value'].sum()

club_values.sort_values(ascending=False).head(5)
# Nationality players count

plt.figure(1, figsize=(22, 12))

plt.title("Which country produce the most players")

sns.countplot(y = "Nationality", order=df['Nationality'].value_counts().index[0:5] ,data=df)

# England is rich at good football players
# Include all the player except goalkeeper

vals = ['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB',

       'LDM', 'CAM', 'CDM', 'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM',

       'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB']

ml_players= df.loc[df['Position'].isin(vals) & df['Position']]
# choose all the columns we need

ml_cols =          ['Crossing', 

                   'Finishing',

                   'HeadingAccuracy',

                   'ShortPassing', 

                   'Volleys', 

                   'Dribbling',

                   'Curve',

                   'FKAccuracy',

                   'LongPassing',

                   'BallControl',

                   'Acceleration',

                    'SprintSpeed',

                   'Agility',

                   'Reactions', 

                   'Balance',

                   'ShotPower', 

                   'Jumping',

                   'Stamina', 

                   'Strength',

                   'LongShots',

                   'Aggression',

                   'Interceptions',

                   'Positioning', 

                   'Vision', 

                   'Penalties',

                   'Composure',

                   'Marking',

                   'StandingTackle', 

                   'SlidingTackle',

                    'Overall'

                   ]
df_ml = pd.DataFrame(data=ml_players, columns=ml_cols)
# check the data

df_ml.info()
df_ml.isnull().any()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# Train test split

y = df_ml['Overall']

X = df_ml[['Crossing', 

           'Finishing',

           'HeadingAccuracy',

           'ShortPassing', 

           'Volleys', 

           'Dribbling',

           'Curve',

           'FKAccuracy',

           'LongPassing',

           'BallControl',

           'Acceleration',

            'SprintSpeed',

           'Agility',

           'Reactions', 

           'Balance',

           'ShotPower', 

           'Jumping',

           'Stamina', 

           'Strength',

           'LongShots',

           'Aggression',

           'Interceptions',

           'Positioning', 

           'Vision', 

           'Penalties',

           'Composure',

           'Marking',

           'StandingTackle', 

           'SlidingTackle']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)
print('Coefficients:', lm.coef_)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted y')
# Evaluate the data

from sklearn import metrics

print("MAE", metrics.mean_absolute_error(y_test, predictions))

print("MSE", metrics.mean_squared_error(y_test, predictions))

print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#  the results seems perfect
sns.distplot((y_test - predictions), bins=50)
# Get the effect of every parameter

coeffecients = pd.DataFrame(lm.coef_, X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients