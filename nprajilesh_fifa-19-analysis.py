# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# regular expressions
import re 

#Counter
from collections import Counter

#setting display options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
df_fifa19 = pd.read_csv(r'../input/data.csv')
df_fifa19.head(5)
# cheking the shape of the data
df_fifa19.shape
#display all columns
df_fifa19.columns
# Selecting only interested columns
chosen_attributes = [
            'Acceleration',
            'Age',
            'Aggression',
            'Agility',
            'Balance',
            'BallControl',
            'Body Type',
            'Club',
            'Composure',
            'Contract Valid Until',
            'Crossing',
            'Curve',
            'Dribbling',
            'Finishing',
            'FKAccuracy',
            'Flag',
            'GKDiving',
            'GKHandling',
            'GKKicking',
            'GKPositioning',
            'GKReflexe',
            'GKReflexes',
            'HeadingAccuracy',
            'Height',
            'Interceptions',
            'International Reputation',
            'Jersey Number',
            'Joined',
            'Jumping',
            'LongPassing',
            'LongShots',
            'Marking',
            'Name',
            'Nationality',
            'Overall',
            'Penalties',
            'Photo',
            'Position',
            'Positioning',
            'Potential',
            'Preferred Foot',
            'Reactions',
            'ShortPassing',
            'ShotPower',
            'Skill Moves',
            'SlidingTackle',
            'Special',
            'SprintSpeed',
            'Stamina',
            'StandingTackle',
            'Strength',
            'Value',
            'Vision',
            'Volleys',
            'Wage',
            'Weak Foot',
            'Weight',
            'Work Rate'
]

df = pd.DataFrame(df_fifa19 , columns = chosen_attributes)
#Transforming Wage and value to numeric

def convert_value(value):
    numeric_val = float(re.findall('\d+\.*\d*', value)[0])
    if 'M' in value:
        numeric_val*= 1000000
    elif 'K' in value:
        numeric_val*= 1000
    return int(numeric_val)
    


#Wage contains data in these formats(€0 ,€100k...)
df['Wage'] = df['Wage'].apply(lambda x: int(re.findall('\d+', x)[0])* 1000)
df['Value'] = df['Value'].apply(convert_value)
plt.figure(1 , figsize = (20 , 13))
corr_columns = ['Acceleration', 'Age', 'Aggression', 'Agility', 'Balance', 'BallControl', 'Crossing', 'Dribbling', 
                'Finishing', 'FKAccuracy', 'HeadingAccuracy', 'Height', 'Interceptions', 'Jumping', 'LongPassing', 
                'LongShots', 'Overall', 'Penalties', 'Positioning', 'Preferred Foot', 'ShortPassing', 'ShotPower', 
                'Skill Moves', 'SprintSpeed', 'Stamina', 'Strength', 'Value', 'Vision', 'Volleys', 'Weak Foot', 
                'Weight', 'Work Rate']
corr_mat = df[corr_columns].corr()
mask = np.zeros_like(corr_mat)
mask[np.triu_indices_from(mask,k=1)] = True
heat_map = sns.heatmap(corr_mat,annot = True, linewidths=.5, cmap='YlOrBr', mask=mask)
heat_map.set_title(label='Heatmap of attributes', fontsize=16)
heat_map
def plot_bar_plot(x = None, y = None, data = None ,hue =None, x_tick_rotation = None ,xlabel = None , 
                  ylabel = None , title = '',ylim = None, palette= None):
    plt.figure(1 , figsize = (15 , 6))
    if x_tick_rotation:
        plt.xticks(rotation = x_tick_rotation)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    sns.barplot(x = x, y = y, hue = hue, data = data, palette= palette)
    
top_30_players = df.sort_values(by = 'Overall' ,ascending = False ).head(30)
plot_bar_plot(x = "Name", y = "Overall", data = top_30_players, palette = 'RdBu', x_tick_rotation= 90 , ylim=(85,96) , title='Top 30 players with highest rating')
top_100_players = df.sort_values(by = 'Overall' ,ascending = False ).head(100)
plt.figure(1 , figsize = (15 , 6))
top_100_players['Age'].plot(kind = 'hist' , bins = 50)
plt.xlabel('Player\'s age')
plt.ylabel('Number of players')
plt.title('Top 100 players Age distribution')
plt.show()
oldest_player_in_top_100 = top_100_players.sort_values(by = 'Age' ,ascending = False ).head(1)[['Name','Age','Overall','Club','Position']]
print(oldest_player_in_top_100)

youngest_player_in_top_100 = top_100_players.sort_values(by = 'Age' ,ascending = True ).head(1)[['Name','Age','Overall','Club','Position']]
print(youngest_player_in_top_100)
df_g = df.groupby(['Club']).sum()[['Overall','Value','Wage']]
df_top_10_club_value = df_g.sort_values(by='Value',ascending=False).head(10)
df_top_10_club_value.reset_index(inplace=True)
df_top_10_club_value
plot_bar_plot(x = "Club", y = "Value", data = df_top_10_club_value, x_tick_rotation= 90 , ylim=(5e8,9e8), palette="Blues_d", title="Club with highest player value")
df_top_10_club_wage = df_g.sort_values(by='Wage',ascending=False).head(10)
df_top_10_club_wage.reset_index(inplace=True)
df_top_10_club_wage
plot_bar_plot(x = "Club", y = "Wage", data = df_top_10_club_wage, x_tick_rotation= 90 , ylim=(1e6,5.5e6), palette="Blues_d", title="Club with highest player Wages")
plt.figure(1 , figsize = (18 , 10))
p1=sns.regplot(data=df_top_10_club_value, x="Value", y="Wage", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':400})
plt.title('Comparison of Players Total wage and Total Value')
for line in range(0,df_top_10_club_value.shape[0]):
     p1.text(df_top_10_club_value.Value[line]+0.2, df_top_10_club_value.Wage[line], df_top_10_club_value.Club[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
countries_with_most_players = [data[0] for data in Counter(df['Nationality']).most_common()[:20]]
df_country = df.loc[df['Nationality'].isin(countries_with_most_players) ]
plt.figure(1 , figsize = (15 , 6))
plt.xticks(rotation = 90)
plt.title('Total Number of players by country')
sns.countplot(x="Nationality", data=df_country, palette="rocket" )
