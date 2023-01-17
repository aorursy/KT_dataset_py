# Data manipulation
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display propertice
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# Date
import datetime

# Maps
import geopandas as gpd
import pycountry

from math import pi

# Display in Jupyter
from IPython.display import display, HTML
# Load dataset
df_fifa19 = pd.read_csv('../input/data.csv')
# Show the first five rows
df_fifa19.head()
# Show the info about dataset
df_fifa19.info()
# Show some statistics about dataset
df_fifa19.describe()
# Shape of dataset (it has 17790 row and 87 columns)
df_fifa19.shape
# Number of unique elements in dataset
df_fifa19.nunique()
# I check where there are NaN values
df_fifa19.isnull().any()
# What columns are in dataset?
df_fifa19.columns
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
# I create DataFrame with chosen columns
df = pd.DataFrame(df_fifa19, columns = chosen_columns)
# The five random rows
df.sample(5)
# Correlation heatmap
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
# Scater plot shows correlation between Acceleration and other chosen features
def make_scatter(df):
    feats = ('Agility', 'Balance', 'Dribbling', 'SprintSpeed')
    
    for index, feat in enumerate(feats):
        plt.subplot(len(feats)/4+1, 4, index+1)
        ax = sns.regplot(x = 'Acceleration', y = feat, data = df)

plt.figure(figsize = (20, 20))
plt.subplots_adjust(hspace = 0.4)

make_scatter(df)
# Histogram: number of players's age
sns.set(style ="dark", palette="colorblind", color_codes=True)
x = df.Age
plt.figure(figsize=(12,8))
ax = sns.distplot(x, bins = 58, kde = False, color='g')
ax.set_xlabel(xlabel="Player\'s age", fontsize=16)
ax.set_ylabel(ylabel='Number of players', fontsize=16)
ax.set_title(label='Histogram of players age', fontsize=20)
plt.show()
# The five eldest players
eldest = df.sort_values('Age', ascending = False)[['Name', 'Nationality', 'Age']].head(3)
eldest.set_index('Name', inplace=True)
print(eldest)
# The five youngest players
eldest = df.sort_values('Age', ascending = True)[['Name', 'Nationality', 'Age']].head(22)
eldest.set_index('Name', inplace=True)
print(eldest)
# Compare six clubs in relation to age
some_clubs = ('Juventus', 'Real Madrid', 'Paris Saint-Germain', 'FC Barcelona', 'Legia Warszawa', 'Manchester United')
df_club = df.loc[df['Club'].isin(some_clubs) & df['Age']]

fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
ax = sns.violinplot(x="Club", y="Age", data=df_club);
ax.set_title(label='Distribution of age in some clubs', fontsize=20);
# The longest membership in the club
now = datetime.datetime.now()
df['Join_year'] = df.Joined.dropna().map(lambda x: x.split(',')[1].split(' ')[1])
df['Years_of_member'] = (df.Join_year.dropna().map(lambda x: now.year - int(x))).astype('int').dropna()
membership = df[['Name', 'Club', 'Years_of_member']].sort_values(by = 'Years_of_member', ascending = False).dropna().head()
membership.set_index('Name', inplace=True)
membership
# The oldest team
df.groupby(['Club'])['Age'].sum().sort_values(ascending = False).head(5)
# The youngest team
df.groupby(['Club'])['Age'].sum().sort_values(ascending = True).head(5)
# The clubs and their players overalls
some_clubs = ('Juventus', 'Real Madrid', 'Paris Saint-Germain', 'FC Barcelona', 'Legia Warszawa', 'Manchester United')
df_club = df.loc[df['Club'].isin(some_clubs) & df['Age'] & df['Overall'] ]

ax = sns.barplot(x=df_club['Club'], y=df_club['Overall'], palette="rocket");
ax.set_title(label='Distribution overall in several clubs', fontsize=20);
# All of position
ax = sns.countplot(x = 'Position', data = df, palette = 'hls');
ax.set_title(label='Count of players on the position', fontsize=20);
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
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(9, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1 
# Top 5 left-footed players
df[df['Preferred Foot'] == 'Left'][['Name','Overall']].head()
# Top 5 right-footed players
df[df['Preferred Foot'] == 'Right'][['Name','Overall']].head()
# Better is left-footed or rigth-footed players?
sns.lmplot(x = 'BallControl', y = 'Dribbling', data = df,
          scatter_kws = {'alpha':0.1},
          col = 'Preferred Foot');
# The clubs, where have players mainly from one country
clubs_coherency = pd.Series()
for club, players in df.groupby(['Club'])['Nationality'].count().items():
    coherency = df[df['Club'] == club].groupby(['Nationality'])['Club'].count().max() / players * 100
    clubs_coherency[club] = coherency

clubs_coherency.sort_values(ascending = False).head(23)
# The clubs with largest number of different countries
df.groupby(['Club'])['Nationality'].nunique().sort_values(ascending = False).head()
# The clubs with the smallest number of foreigners players
df.groupby(['Club'])['Nationality'].nunique().sort_values().head()
# Relation dribbling and crossing with respected finishing of players
plt.figure(figsize=(14,7))
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

ax = sns.scatterplot(x='Crossing', y='Dribbling',
                     hue='Finishing',
                     palette=cmap, sizes=(1, 1),
                     data=df)
ax.set_title(label='Relation dribbling and crossing with respected finishing of players', fontsize=20);
# Relation stamina and age with respected sprint speed of players
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

ax = sns.scatterplot(x='Age', y='Stamina',
                     hue='SprintSpeed',
                     palette=cmap, sizes=(1, 1),
                     data=df)
ax.set_title(label='Relation stamina and age with respected sprint speed of players', fontsize=20);
# Crossing vs. dribbling
sns.jointplot(x=df['Dribbling'], y=df['Crossing'], kind="hex", color="#4CB391");
# The value has some non numeric mark so I extract rigth value
def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

df['Value_float'] = df['Value'].apply(value_to_int)
# Top five the most expensive clubs
df.groupby(['Club'])['Value_float'].sum().sort_values(ascending = False).head(5)
# Top five the less expensive clubs
df.groupby(['Club'])['Value_float'].sum().sort_values().head(5)
# Top five teams with the best players
df.groupby(['Club'])['Overall'].max().sort_values(ascending = False).head()
# Value vs. Overall
value = df.Value_float
ax = sns.regplot(x = value / 10000000, y = 'Overall', fit_reg = False, data = df);
ax.set_title(label='Value vs. Overall', fontsize=20);
# Relation potential and age with respected value of players
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

sns.relplot(x="Age", y="Potential", hue=value/100000, 
            sizes=(40, 400), alpha=.5,
            height=6, data=df);