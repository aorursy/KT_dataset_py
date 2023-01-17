# Data and arrays handling
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive plotting
from plotly.offline import init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)

# Inline plots
%matplotlib inline

# Ignore warns
import warnings
warnings.filterwarnings('ignore')
%%time

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(3).T
train.info()
plt.figure(figsize=(8,6))
sns.distplot(train['assists'], kde=False)
plt.show()
assist_df = train[['assists', 'winPlacePerc']]
assist_df['assists'] = assist_df['assists'].apply(lambda x: 'zero' if x == 0 else '1 or more')
assist_df['assists'].value_counts(normalize=True)
plt.figure(figsize=(8,7))
sns.boxplot(x='assists', y='winPlacePerc', data=assist_df)
plt.title('Zero assists vs. one or more assists')
plt.show()
plt.figure(figsize=(8,6))
sns.distplot(train['boosts'], kde=False)
plt.show()
boosts_df = train[['boosts', 'winPlacePerc']]
boosts_df['boosts'] = boosts_df['boosts'].apply(lambda x: 'zero' if x == 0 else '1 or more')
boosts_df['boosts'].value_counts()
plt.figure(figsize=(8,7))
sns.boxplot(x='boosts', y='winPlacePerc', data=boosts_df, order=['zero', '1 or more'])
plt.show()
attrs_for_boxplots = ['damageDealt', 'DBNOs',
                       'headshotKills', 'heals', 'kills',
                       'killStreaks', 'longestKill', 'revives',
                       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
                       'vehicleDestroys', 'walkDistance', 'weaponsAcquired']

other_attr = ['killPlace', 'killPoints', 'maxPlace', 'winPoints', 'numGroups']

target = 'winPlacePerc'
%%time

# Number of columns in our big picture
columns = 2

# Number of rows
rows = len(attrs_for_boxplots)

# Position index
plot_position_in_grid = 1

# Iterate through all attributes
for attr in attrs_for_boxplots:
    
    # Set figure size
    plt.figure(figsize=(12, 4 * rows))
    
    # fix the subplot position
    # plot the distribution
    plt.subplot(rows, columns, plot_position_in_grid)
    sns.distplot(train[attr], kde=False)
    
    # Create compare df
    temp_df = train[[attr, target]]
    temp_df[attr] = temp_df[attr].apply(lambda x: 'zero' if x == 0 else 'more')

    # fix the subplot position
    # plot the boxplot
    plt.subplot(rows, columns, plot_position_in_grid+1)
    sns.boxplot(x=attr, y=target, data=temp_df, order=['zero', 'more'])

    plot_position_in_grid += 2
%%time
corr_matrix = train.corr()

plt.figure(figsize=(24,23))
sns.set(font_scale=1.3)
sns.heatmap(corr_matrix, annot=True, fmt='.1f')
plt.show()
corr_with_target = train.drop(['Id', 'matchId', 'groupId', 'winPlacePerc'], 1).corrwith(train['winPlacePerc'])
# Set our dots
trace0 = go.Scatter(
    x = corr_with_target.index,
    y = corr_with_target.values,
    name = 'corrs',
    mode='markers',
    marker = {
        'size' : 20,
        'color' : corr_with_target.values,
        'colorscale' : 'Jet',
        'showscale' : True,
        'symbol' : 202,
        'opacity' : .76
    }
)

# Create data array and layout
data = [trace0]
layout = {'title': 'Correlation between winPlacePerc and other attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

# Display it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
plt.figure(figsize=(8,6))
plt.scatter(train['killPlace'], train['damageDealt'], alpha=.8, c='orange')
plt.xlabel('killPlace')
plt.ylabel('damageDealth')
plt.title('killPlace vs damageDealt scatter plot')
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(train['heals'], train['boosts'], alpha=.8, c='green')
plt.xlabel('heals')
plt.ylabel('boosts')
plt.title('Heals vs boots scatter plot')
plt.show()
def get_subplot_row_columns(df, corr_threshold):
    corr = df.corr()
    n = corr[(corr > corr_threshold) & (corr != 1)].dropna(axis=0, how='all') \
                                           .dropna(axis=1, how='all') \
                                           .notnull().sum().sum()
    cols = 3

    return cols, int(np.ceil(n / cols))
%%time
# It takes some time to display all graphs

# Feel free to play with this value
threshold = .5

# All attributes
attrs = train.columns.values

# Array for pairs already checked
seen_pairs = []

# Params for subplotting
subplot_number = 0
cols, rows = get_subplot_row_columns(train, threshold)

fig, axs = plt.subplots(rows, cols, figsize=(18, 5 * rows))
plt.subplots_adjust(wspace=0.45, hspace=0.35)
axs = axs.ravel()

print(f'{rows} x {cols}')

for first_attr in attrs:
    for second_attr in attrs:       
        # Skip same
        if first_attr == second_attr:
            continue
        
        # Skip swap attributes
        if (first_attr, second_attr) in seen_pairs:
            continue
        else:
            seen_pairs.append((second_attr, first_attr))
        
        # Check threshold
        current_corr = train[[first_attr]].corrwith(train[second_attr])[first_attr]
        
        if current_corr > threshold:  
            print(f'{subplot_number} - {first_attr} x {second_attr}')
            axs[subplot_number].set_title(f'{first_attr} vs {second_attr}\n(correlation = {current_corr})')
            axs[subplot_number].scatter(train[first_attr], train[second_attr], alpha=.8, c='green')
            axs[subplot_number].set_xlabel(first_attr)
            axs[subplot_number].set_ylabel(second_attr)
            
            
            subplot_number += 1

            
for i in range(subplot_number, cols*rows):
    fig.delaxes(axs[i])
    
print(f'{subplot_number} graphs plotted')
plt.figure(figsize=(8,6))
sns.lineplot(x='vehicleDestroys', y='winPlacePerc',
             data=train)
plt.title('vehicleDestroys impact on winPlacePerc')
solos = train[train['numGroups']>50]
duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]
squads = train[train['numGroups']<=25]
games_count = train.shape[0]


game_types_df = pd.DataFrame({'Games count' : [solos.shape[0],
                                         duos.shape[0],
                                         squads.shape[0]],
                              'Normalized' : [solos.shape[0] / games_count,
                                         duos.shape[0] / games_count,
                                         squads.shape[0] / games_count]},
                              index=['solos', 'duos', 'squads'])
game_types_df
plt.figure(figsize=(18,8))

sns.pointplot(x='kills', y='winPlacePerc', data=solos, color='green')
sns.pointplot(x='kills', y='winPlacePerc', data=duos, color='red')
sns.pointplot(x='kills', y='winPlacePerc', data=squads, color='black')

plt.text(14,0.5,'Solos',color='green',fontsize = 17,style = 'italic')
plt.text(14,0.45,'Duos',color='red',fontsize = 17,style = 'italic')
plt.text(14,0.40,'Squads',color='black',fontsize = 17,style = 'italic')

plt.title('Kills in different play modes impact pn winPlacePerc')

plt.grid()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
data = train.copy()
data = data[data['playersJoined']>49]
plt.figure(figsize=(18,9))
sns.countplot(data['playersJoined'])
plt.title("Players Joined",fontsize=15)
plt.show()
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

train[['playersJoined', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm']].head(4)
train['healsAndBoosts'] = train['heals']+train['boosts']
train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']
train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
train['boostsPerWalkDistance'].fillna(0, inplace=True)
train['healsPerWalkDistance'] = train['heals']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
train['healsPerWalkDistance'].fillna(0, inplace=True)
train['healsAndBoostsPerWalkDistance'] = train['healsAndBoosts']/(train['walkDistance']+1) #The +1 is to avoid infinity.
train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
train[['walkDistance', 'boosts', 'boostsPerWalkDistance' ,'heals',  'healsPerWalkDistance', 'healsAndBoosts', 'healsAndBoostsPerWalkDistance']][40:45]
train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.
train['killsPerWalkDistance'].fillna(0, inplace=True)
train[['kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance', 'winPlacePerc']].sort_values(by='killsPerWalkDistance').tail(10)
train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]