import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/data.csv")
df.head()
df.describe()
df.info()
df.columns
plt.rcParams['figure.figsize']=(25,16)

hm=sns.heatmap(df[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(), annot = True, linewidths=.5, cmap='Blues')

hm.set_title(label='Heatmap of dataset', fontsize=20)

hm;
def make_scatter(df):

    feats = ('Agility', 'Balance', 'Dribbling', 'SprintSpeed')

    

    for index, feat in enumerate(feats):

        plt.subplot(len(feats)/4+1, 4, index+1)

        ax = sns.regplot(x = 'Acceleration', y = feat, data = df)



plt.figure(figsize = (20, 20))

plt.subplots_adjust(hspace = 0.4)



make_scatter(df)
# The five eldest players

eldest = df.sort_values('Age', ascending = False)[['Name', 'Nationality', 'Age']].head(3)

eldest.set_index('Name', inplace=True)

print(eldest)
sns.violinplot(x="Club", y="Age", data=df)
# Better is left-footed or rigth-footed players?

sns.lmplot(x = 'BallControl', y = 'Dribbling', data = df,

          scatter_kws = {'alpha':0.1},

          col = 'Preferred Foot')