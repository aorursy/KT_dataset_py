# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns               # Provides a high level interface for drawing attractive and informative statistical graphics

%matplotlib inline

sns.set()

from subprocess import check_output



import warnings                                            # Ignore warning related to pandas_profiling

warnings.filterwarnings('ignore') 



def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

         ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='black', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

         ax.annotate(f"{p.get_height() * 100 / new_df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='black', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

            

import os

print(os.listdir("../input"))



df = pd.read_csv('../input/data.csv')
df.info()
df.isnull().sum()
df.head()
df.nunique() #unique elements in dataset
# I choose interesting to me columns. Later I will use them for analysis.

chosen_columns = [

    'Name',

    'Age',

    'Nationality',

    'Overall',

    'Potential',

    'Special',

    'Acceleration',

    'Aggression',

    'Agility',

    'Balance',

    'BallControl',

    'Body Type',

    'Composure',

    'Crossing',

    'Curve',

    'Club',

    'Dribbling',

    'FKAccuracy',

    'Finishing',

    'GKDiving',

    'GKHandling',

    'GKKicking',

    'GKPositioning',

    'GKReflexes',

    'HeadingAccuracy',

    'Interceptions',

    'International Reputation',

    'Jersey Number',

    'Jumping',

    'Joined',

    'LongPassing',

    'LongShots',

    'Marking',

    'Penalties',

    'Position',

    'Positioning',

    'Preferred Foot',

    'Reactions',

    'ShortPassing',

    'ShotPower',

    'Skill Moves',

    'SlidingTackle',

    'SprintSpeed',

    'Stamina',

    'StandingTackle',

    'Strength',

    'Value',

    'Vision',

    'Volleys',

    'Wage',

    'Weak Foot',

    'Work Rate'

]
new_df = pd.DataFrame(df, columns = chosen_columns)
new_df.head()
df['Body Type'].unique()
df['Body Type'].fillna('Normal', inplace = True)
df['Body Type'].replace(to_replace = ['Messi','C. Ronaldo','Neymar', 'PLAYER_BODY_TYPE_25', 'Shaqiri','Akinfenwa','Courtois'], 

                        value = ['Normal','Normal','Normal','Normal','Normal','Normal','Normal'],inplace = True)
new_df.nunique()
new_df.describe()
corr_df = new_df[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']]



corr = corr_df.corr()

plt.figure(figsize = (25,16))

heatmap = sns.heatmap(corr,annot = True, linewidths=.5)

heatmap.set_title(label='Heatmap of dataset', fontsize=25)

plt.show()
plt.figure(figsize=(8,5))

ax = sns.countplot('Body Type', data = df)

plt.ylabel('Number of players')

annot_plot(ax,0.08,1)

plt.figure(figsize=(8,5))

ax = sns.countplot('International Reputation', data = df)

plt.ylabel('Number of players')

annot_plot(ax,0.08,1)

plt.figure(figsize=(8,5))

ax = sns.countplot('Preferred Foot', data = df)

plt.ylabel('Number of players')

annot_plot(ax,0.08,1)
plt.figure(figsize=(8,5))

ax = sns.countplot('Skill Moves', data = df)

plt.ylabel('Number of players')

annot_plot(ax,0.08,1)
plt.figure(figsize=(8,5))

ax = sns.countplot('Weak Foot', data = df)

plt.ylabel('Number of players')

annot_plot(ax,0.08,1)
plt.figure(figsize=(8,5))

ax = sns.countplot('Work Rate', data = df)

plt.ylabel('Number of players')

plt.xticks(rotation = 45, ha = 'right')

annot_plot(ax,0.08,1)
def num_annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

plt.figure(figsize=(22,12))

ax = sns.countplot('Age', data = df,palette='viridis')

plt.ylabel('Number of players')

plt.xticks(rotation = 45, ha = 'right')

num_annot_plot(ax,0.08,1)
plt.figure(figsize = (14,7))

ax = sns.countplot('Nationality', data = new_df ,order = new_df['Nationality'].value_counts()[:20].index,palette='viridis')

plt.ylabel('Number of Players.')

plt.xticks(rotation = 45, ha = 'right')

num_annot_plot(ax,0.08,1)

plt.show()
# The five eldest players

eldest = df.sort_values('Age', ascending = False)[['Name', 'Nationality', 'Age']].head(3)

eldest.set_index('Name', inplace=True)

print(eldest)
# The five youngest players

eldest = new_df.sort_values('Age', ascending = True)[['Name', 'Nationality', 'Age']].head(22)

eldest.set_index('Name', inplace=True)

print(eldest)
plt.figure(figsize = (18,10))

ax = sns.countplot('Position', data = new_df ,order = new_df['Position'].value_counts().index,palette='viridis')

plt.ylabel('Number of Players.')

plt.xticks(rotation = 45, ha = 'right')

num_annot_plot(ax,0.08,1)

plt.show()
# The best player per position

display(HTML(df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Name', 'Position']].to_html(index=False)))
player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

)



# Top three features per position

for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(3).index)))
# Top 5 right-footed players

df[df['Preferred Foot'] == 'Right'][['Name','Overall']].head()
# Top 5 LEFT-footed players

df[df['Preferred Foot'] == 'Left'][['Name','Overall']].head()