# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sns.set(style= 'darkgrid')

player_df = pd.read_csv('../input/data.csv')

numcols = ['Overall', 'Potential', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 

          'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',  'Stamina','Value','Wage']

catcols = ['Name','Club','Nationality','Preferred Foot','Position','Body Type']

player_df = player_df[numcols + catcols]

player_df.head(5)

def wage_split(x):

    return int(x.split('K')[0][1:])

def value_split(x):

    if 'M' in x:

        return float(x.split('M')[0][1:])

    elif 'K' in x:

        return float(x.split('K')[0][1:])/1000

    
player_df['Wage'] = player_df['Wage'].apply(lambda x: wage_split(x))

player_df['Value'] = player_df['Value'].apply(lambda x: value_split(x))
correlation = player_df.corr()

p = sns.heatmap(correlation, vmax= 0.5, center= 0, square=True, linewidths= 0.3, annot= True, fmt = '.2f')

sns.despine()

p.figure.set_size_inches(14, 10)

plt.show()
sns.set(style= 'ticks')

filtered_player_df = player_df[(player_df['Club'].isin(['FC Barcelona', 'Chelsea', 'Manchester United', 'Manchester City', 

                                                       'Paris Saint-Germain', 'Real Madrid', 'FC Bayern München', 'FC Porto']))

                              & (player_df['Nationality'].isin(['Brazil', 'Germany', 'Italy', 'England', 'Spain','Argentina']))

                              ]

df_plot = filtered_player_df.groupby(['Club', 'Nationality']).size().reset_index().pivot(columns='Club', index='Nationality', values=0).reset_index()

p = df_plot.set_index('Nationality').T.plot(kind='bar', stacked=True)

sns.despine()

p.figure.set_size_inches(14,8)

    

plt.show()

filtered_player_df = player_df[(player_df['Club'].isin(['FC Barcelona', 'Chelsea', 'Manchester United', 'Manchester City', 

                                                       'Paris Saint-Germain', 'Real Madrid', 'FC Bayern München', 'FC Porto']))

                              & (player_df['Nationality'].isin(['Brazil', 'Germany', 'Italy', 'England', 'Spain','Argentina']))

                              ]

p = sns.pairplot(filtered_player_df[['Value', 'SprintSpeed', 'Potential', 'Wage']])
p = sns.pairplot(filtered_player_df[['Value', 'SprintSpeed', 'Potential', 'Wage', 'Club']], hue= 'Club')
p = sns.swarmplot(y = 'Club', x = 'Wage', data = filtered_player_df, size = 7)

sns.despine()

p.figure.set_size_inches(14,10)

plt.show()
g = sns.boxplot(y = "Club",

              x = 'Wage', 

              data = filtered_player_df, whis=np.inf)

g = sns.swarmplot(y = "Club",

              x = 'Wage', 

              data = filtered_player_df,

              # Decrease the size of the points to avoid crowding 

              size = 7,color = 'black')

# remove the top and right line in graph

sns.despine()

g.figure.set_size_inches(12,8)

plt.show()
max_wage = filtered_player_df.Wage.max()

max_wage_player = filtered_player_df[(player_df['Wage'] == max_wage)]['Name'].values[0]

g = sns.boxplot(y = "Club",

              x = 'Wage', 

              data = filtered_player_df, whis=np.inf)

g = sns.swarmplot(y = "Club",

              x = 'Wage', 

              data = filtered_player_df,

              size = 7,color='black')

sns.despine()

plt.annotate(s = max_wage_player,

             xy = (max_wage,0),

             xytext = (500,1), 

             # Shrink the arrow to avoid occlusion

             arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03},

             backgroundcolor = 'white')

g.figure.set_size_inches(12,8)

plt.show()