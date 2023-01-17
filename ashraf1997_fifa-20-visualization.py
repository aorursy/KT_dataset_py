#image display
from IPython.display import Image
Image("../input/fifa20image/960.jpg")
# import libraries for basic operations
import numpy as np
import pandas as pd 

# import libraries for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
# import file and show computation time
%time data = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')
data.shape
data.info()
data.head()
data.describe()
# view particular column in the dataset
data['club'].head()
# number of clubs in the dataset
len(data['club'].unique())
# view clubs in the dataset
print(data['club'].unique())
# view all columns
for col in data.columns:
    print(col)
# view particular club data
def club(x):
    return data[data['club'] == x][['short_name','team_jersey_number','player_positions','overall','nationality','age','wage_eur',
                                    'value_eur','contract_valid_until']]

club('FC Barcelona')
# column shape
x = club('FC Barcelona')
x.shape
# number of players in club
print(len(club('FC Barcelona')))
# minimum of a column
data['release_clause_eur'].min() #max,mean
# checking if the data contains any NULL value and printing to list
data_na=data.columns[data.isna().any()].tolist()
print("List of cloumns with missing values","\n\n",data_na)
# printing sum of missing values
data.isnull().sum()
# visualizing missing values
allna = (data.isnull().sum() / len(data))*100
allna = allna.drop(allna[allna == 0].index).sort_values()
plt.figure(figsize=(15, 10))
allna.plot.barh(color=('red', 'black'), edgecolor='black')
plt.title('Missing values percentage per column', fontsize=15, weight='bold' )
plt.xlabel('Percentage', size=15)
plt.ylabel('Features with missing values')
plt.yticks()
plt.show()
# grouping missing values into categorical and numerical value 
NA=data[['release_clause_eur','player_tags','team_position','team_jersey_number',
'loaned_from','joined','contract_valid_until','nation_position','nation_jersey_number','pace',
'shooting','passing','dribbling','defending','physic',
'gk_diving','gk_handling','gk_kicking','gk_reflexes','gk_speed','gk_positioning','player_traits','ls','st','rs','lw',
'lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb']]
NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print(NAcat.shape[1],'categorical features with missing values')
print(NAnum.shape[1],'numerical features with missing values')
# sum of Na value in a column before filling
data['release_clause_eur'].isnull().sum()
# filling the missing value for the continous variables for proper data visualization
data['release_clause_eur'].fillna(0,inplace=True)
data['player_tags'].fillna('#Team Player',inplace=True)
data['team_position'].fillna('Not Decided',inplace=True)                                  
data['team_jersey_number'].fillna(0,inplace=True)
data['loaned_from'].fillna('Disclosed',inplace=True)
data['joined'].fillna('Disclosed',inplace=True)
data['contract_valid_until'].fillna('Disclosed',inplace=True)
data['nation_position'].fillna('Not Decided',inplace=True)
data['nation_jersey_number'].fillna(0,inplace=True)
data['pace'].fillna(data['pace'].mean(),inplace=True)
data['shooting'].fillna(data['shooting'].mean(),inplace=True)
data['passing'].fillna(data['passing'].mean(),inplace=True)
data['dribbling'].fillna(data['dribbling'].mean(),inplace=True)
data['defending'].fillna(data['defending'].mean(),inplace=True)
data['physic'].fillna(data['physic'].mean(),inplace=True)
data['gk_diving'].fillna(data['gk_diving'].mean(),inplace=True)
data['gk_handling'].fillna(data['gk_handling'].mean(),inplace=True)
data['gk_kicking'].fillna(data['gk_kicking'].mean(),inplace=True)
data['gk_reflexes'].fillna(data['gk_reflexes'].mean(),inplace=True)
data['gk_speed'].fillna(data['gk_speed'].mean(),inplace=True)
data['gk_positioning'].fillna(data['gk_positioning'].mean(),inplace=True)
data['player_traits'].fillna('Not Analyzed',inplace=True)
data['ls'].fillna('Not Analyzed',inplace=True)
data['st'].fillna('Not Analyzed',inplace=True)
data['rs'].fillna('Not Analyzed',inplace=True)
data['lw'].fillna('Not Analyzed',inplace=True)
data['lf'].fillna('Not Analyzed',inplace=True)
data['cf'].fillna('Not Analyzed',inplace=True)
data['rf'].fillna('Not Analyzed',inplace=True)
data['rw'].fillna('Not Analyzed',inplace=True)
data['lam'].fillna('Not Analyzed',inplace=True)
data['cam'].fillna('Not Analyzed',inplace=True)
data['ram'].fillna('Not Analyzed',inplace=True)
data['lm'].fillna('Not Analyzed',inplace=True)
data['lcm'].fillna('Not Analyzed',inplace=True)
data['cm'].fillna('Not Analyzed',inplace=True)
data['rcm'].fillna('Not Analyzed',inplace=True)
data['rm'].fillna('Not Analyzed',inplace=True)
data['lwb'].fillna('Not Analyzed',inplace=True)
data['ldm'].fillna('Not Analyzed',inplace=True)
data['cdm'].fillna('Not Analyzed',inplace=True)
data['rdm'].fillna('Not Analyzed',inplace=True)
data['rwb'].fillna('Not Analyzed',inplace=True)
data['lb'].fillna('Not Analyzed',inplace=True)
data['lcb'].fillna('Not Analyzed',inplace=True)
data['cb'].fillna('Not Analyzed',inplace=True)
data['rcb'].fillna('Not Analyzed',inplace=True)
data['rb'].fillna('Not Analyzed',inplace=True)
data.fillna(0, inplace = True)
# view sum of Na value in column after filling
data['release_clause_eur'].isnull().sum()
# checking if the data contains any NULL value after filling Na values
data_na_after=data.columns[data.isna().any()].tolist()
data_na_after
# functions to categorize features
def defending(data):
    return int(round((data[['defending_marking', 'defending_standing_tackle', 
                               'defending_sliding_tackle']].mean()).mean()))
def mental(data):
    return int(round((data[['mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 
                               'mentality_vision','mentality_composure']].mean()).mean()))
def passing(data):
    return int(round((data[['attacking_crossing', 'attacking_short_passing', 
                               'skill_long_passing']].mean()).mean()))
def mobility(data):
    return int(round((data[['movement_acceleration', 'movement_sprint_speed', 
                               'movement_agility','movement_reactions']].mean()).mean()))
def power(data):
    return int(round((data[['movement_balance', 'power_jumping', 'power_stamina', 
                               'power_strength']].mean()).mean()))
def rating(data):
    return int(round((data[['potential', 'overall']].mean()).mean()))
def shooting(data):
    return int(round((data[['attacking_finishing', 'attacking_volleys', 'skill_fk_accuracy', 
                               'power_shot_power','power_long_shots', 'mentality_penalties']].mean()).mean()))
# adding newly created categories(columns) to the dataset

data['cat_defending'] = data.apply(defending, axis = 1)
data['cat_mental'] = data.apply(mental, axis = 1)
data['cat_passing'] = data.apply(passing, axis = 1)
data['cat_mobility'] = data.apply(mobility, axis = 1)
data['cat_power'] = data.apply(power, axis = 1)
data['cat_rating'] = data.apply(rating, axis = 1)
data['cat_shooting'] = data.apply(shooting, axis = 1)
# creating sub dataframe using the newly created columns

players = data[['short_name','cat_defending','cat_mental','cat_passing',
                'cat_mobility','cat_power','cat_rating','cat_shooting','age',
                'nationality', 'club']]

players.head()
# visualize comparison of preferred foot over the different players
plt.rcParams['figure.figsize'] = (18, 8)
sns.countplot(data['preferred_foot'], palette = 'pink')
plt.title('Most Preferred Foot of the Players', fontsize = 20)
plt.show()
# plotting a pie chart to represent share of international repuatation
labels = ['1', '2', '3', '4', '5']
sizes = data['international_reputation'].value_counts()
colors = plt.cm.copper(np.linspace(0, 1, 5))
explode = [0.1, 0.1, 0.2, 0.5, 0.9]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)
plt.title('International Repuatation for the Football Players', fontsize = 20)
plt.legend()
plt.show()
# visualize different team positions acquired by the players 
plt.figure(figsize = (18, 8))
plt.style.use('fivethirtyeight')
ax = sns.countplot('team_position', data = data, palette = 'bone')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.show()
# visualize Players' Wages
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (15, 5)
sns.distplot(data['wage_eur'], color = 'blue')
plt.xlabel('Wage Range for Players', fontsize = 16)
plt.ylabel('Count of the Players', fontsize = 16)
plt.title('Distribution of Wages of Players', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
# visualize Skill Moves of Players
plt.figure(figsize = (10, 8))
ax = sns.countplot(x = 'skill_moves', data = data, palette = 'pastel')
ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)
ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()
# visualize Height of Players
plt.figure(figsize = (25,8))
ax = sns.countplot(x = 'height_cm', data = data, palette = 'dark')
ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)
ax.set_xlabel(xlabel = 'Height(cm)', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()
# visualize Weight of Players
plt.figure(figsize = (25,8))
ax = sns.countplot(x = 'weight_kg', data = data, palette = 'dark')
ax.set_title(label = 'Count of players on Basis of Weight', fontsize = 20)
ax.set_xlabel(xlabel = 'Weight(kg)')
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()
# visualize Different Work rate of the players participating in the FIFA 2020

plt.figure(figsize = (15, 7))
plt.style.use('_classic_test_patch')

sns.countplot(x = 'work_rate', data = data, palette = 'hls')
plt.title('Different work rates of the Players Participating in the FIFA 2020', fontsize = 20)
plt.xlabel('Work rates associated with the players', fontsize = 16)
plt.ylabel('Count of players', fontsize = 16)
plt.show()
# visualize Different potential scores of the players participating in the FIFA 2019

x = data.potential
plt.figure(figsize=(12,8))
plt.style.use('seaborn-paper')

ax = sns.distplot(x, bins = 58, kde = False, color = 'y')
ax.set_xlabel(xlabel = "Player\'s Potential Scores", fontsize = 16)
ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)
ax.set_title(label = 'Histogram of players Potential Scores', fontsize = 20)
plt.show()
# visualize Different overall scores of the players participating in the FIFA 2020

sns.set(style = "dark", palette = "deep", color_codes = True)
x = data.overall
plt.figure(figsize = (12,8))
plt.style.use('ggplot')

ax = sns.distplot(x, bins = 52, kde = False, color = 'r')
ax.set_xlabel(xlabel = "Player\'s Scores", fontsize = 16)
ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)
ax.set_title(label = 'Histogram of players Overall Scores', fontsize = 20)
plt.show()
# visualize Different nations participating in the FIFA 2020

plt.style.use('dark_background')
data['nationality'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))
plt.title('Different Nations Participating in FIFA 2020', fontsize = 30, fontweight = 20)
plt.xlabel('Name of The Country')
plt.ylabel('count')
plt.show()
# visualize Players' age

sns.set(style = "dark", palette = "colorblind", color_codes = True)
x = data.age
plt.figure(figsize = (15,8))
ax = sns.distplot(x, bins = 58, kde = False, color = 'g')
ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)
ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)
ax.set_title(label = 'Histogram of players\' age', fontsize = 20)
plt.show()
# visualize violin plot 

plt.rcParams['figure.figsize'] = (20, 7)
plt.style.use('seaborn-dark-palette')

sns.boxenplot(data['overall'], data['age'], hue = data['preferred_foot'], palette = 'Greys')
plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)
plt.show()
# visualize bubble plot

plt.rcParams['figure.figsize'] = (20, 7)
plt.scatter(data['overall'], data['international_reputation'], s = data['age']*1000, c = 'pink')
plt.xlabel('Overall Ratings', fontsize = 20)
plt.ylabel('International Reputation', fontsize = 20)
plt.title('Ratings vs Reputation', fontweight = 20, fontsize = 20)
#plt.legend('Age', loc = 'upper left')
plt.show()
# selecting some of the interesting and important columns from the set of columns in the given dataset

selected_columns = ['short_name', 'age', 'nationality', 'overall', 'potential', 'club', 'value_eur',
                    'wage_eur', 'preferred_foot', 'international_reputation', 'weak_foot',
                    'skill_moves', 'work_rate', 'body_type', 'team_position', 'height_cm', 'weight_kg ',
                    'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
                    'attacking_volleys', 'skill_dribbling','movement_acceleration','power_shot_power',
                    'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 
                    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 
                    'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
                    'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 
                    'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving',
                    'goalkeeping_handling',
                    'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'release_clause_eur','mentality_composure']
data_selected = pd.DataFrame(data, columns = selected_columns)
data_selected.columns
# having a look at the sample of selected data

data_selected.sample(5)
# plotting a correlation heatmap

plt.rcParams['figure.figsize'] = (30, 20)
sns.heatmap(data_selected[['short_name', 'age', 'nationality', 'overall', 'potential', 'club', 'value_eur',
                    'wage_eur', 'preferred_foot', 'international_reputation', 'weak_foot',
                    'skill_moves', 'work_rate', 'body_type', 'team_position', 'height_cm',
                    'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
                    'attacking_volleys', 'skill_dribbling','power_shot_power',
                    'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 
                    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 
                    'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
                    'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 
                    'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving',
                    'goalkeeping_handling',
                    'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'release_clause_eur','mentality_composure']].corr(), annot = True)
plt.title('Correlation between features', fontsize = 30)
plt.show()
# best players per each position with their age, club, and nationality based on their overall scores

data.iloc[data.groupby(data['team_position'])['overall'].idxmax()][['team_position', 'short_name', 'age', 'club', 'nationality']]
# best players from each positions with their age, nationality, club based on their potential scores

data.iloc[data.groupby(data['team_position'])['potential'].idxmax()][['team_position', 'short_name', 'age', 'club', 'nationality']]
# picking up the countries with highest number of players to compare their overall scores

data['nationality'].value_counts().head(8)
# Every Nations' Player and their Weights

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')
data_countries = data.loc[data['nationality'].isin(some_countries) & data['weight_kg']]

plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.violinplot(x = data_countries['nationality'], y = data_countries['weight_kg'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Weight in kg', fontsize = 9)
ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)
plt.show()
# Every Nations' Player and their overall scores

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')
data_countries = data.loc[data['nationality'].isin(some_countries) & data['overall']]

plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.barplot(x = data_countries['nationality'], y = data_countries['overall'], palette = 'spring')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 9)
ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)
plt.show()
# Every Nations' Player and their International Reputation

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')
data_countries = data.loc[data['nationality'].isin(some_countries) & data['international_reputation']]

plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.boxenplot(x = data_countries['nationality'], y = data_countries['international_reputation'], palette = 'autumn')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Distribution of reputation', fontsize = 9)
ax.set_title(label = 'Distribution of International Repuatation of players from different countries', fontsize = 15)
plt.show()
# finding the the popular clubs around the globe
data['club'].value_counts().head(10)
# visualize distribution of Overall Score in Different popular Clubs
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',
             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_clubs = data.loc[data['club'].isin(some_clubs) & data['overall']]

plt.rcParams['figure.figsize'] = (15, 8)
ax = sns.boxplot(x = data_clubs['club'], y = data_clubs['overall'], palette = 'autumn')
ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)
ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)
ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
# Distribution of international reputation in some Popular clubs

some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',
             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_club = data.loc[data['club'].isin(some_clubs) & data['international_reputation']]

plt.rcParams['figure.figsize'] = (16, 8)
ax = sns.violinplot(x = 'club', y = 'international_reputation', data = data_club, palette = 'autumn')
ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 10)
ax.set_ylabel(ylabel = 'Distribution of Reputation', fontsize = 10)
ax.set_title(label = 'Disstribution of International Reputation in some Popular Clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
#distribution in weights in popular clubs
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',
             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_clubs = data.loc[data['club'].isin(some_clubs) & data['weight_kg']]

plt.rcParams['figure.figsize'] = (15, 8)
ax = sns.violinplot(x = 'club', y = 'weight_kg', data = data_clubs, palette = 'PuBu')
ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)
ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)
ax.set_title(label = 'Distribution of Weight in Different popular Clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()
# finding 15 youngest Players from the dataset

youngest = data.sort_values('age', ascending = True)[['short_name', 'age', 'club', 'nationality']].head(15)
print(youngest)
# finding 15 eldest players from the dataset

eldest = data.sort_values('age', ascending = False)[['short_name', 'age', 'club', 'nationality']].head(15)
print(eldest)
# checking the head of the joined column

data['joined'].head()
# defining the features of players

player_features = ('movement_acceleration', 'mentality_aggression', 'movement_agility', 
                   'movement_balance', 'skill_ball_control', 'mentality_composure', 
                   'attacking_crossing', 'skill_dribbling', 'skill_fk_accuracy', 
                   'attacking_finishing', 'gk_diving', 'gk_handling', 
                   'gk_kicking', 'gk_positioning', 'gk_reflexes', 
                   'attacking_heading_accuracy', 'mentality_interceptions', 'power_jumping', 
                   'skill_long_passing', 'power_long_shots', 'defending_marking', 'mentality_penalties')

# Top four features for every position in football

for i, val in data.groupby(data['player_positions'])[player_features].mean().iterrows():
    print('player_positions {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
from math import pi

idx = 1
plt.figure(figsize=(15,45))
for position_name, features in data.groupby(data['player_positions'])[player_features].mean().iterrows():
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
# Top 10 left footed footballers

data[data['preferred_foot'] == 'Left'][['short_name', 'age', 'club', 'nationality']].head(10)
# Top 10 Right footed footballers

data[data['preferred_foot'] == 'Right'][['short_name', 'age', 'club', 'nationality']].head(10)
# comparing the performance of left-footed and right-footed footballers
# ballcontrol vs dribbing

sns.lmplot(x = 'skill_ball_control', y = 'skill_dribbling', data = data, col = 'preferred_foot')
plt.show()
# visualizing clubs with highest number of different countries

data.groupby(data['club'])['nationality'].nunique().sort_values(ascending = False).head(10)
# visualizing clubs with highest number of different countries

data.groupby(data['club'])['nationality'].nunique().sort_values(ascending = True).head(10)
sns.lineplot(data['age'], data['cat_rating'], palette = 'Wistia')
plt.title('Age vs Rating', fontsize = 20)

plt.show()
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)
import plotly.graph_objs as go
rating = pd.DataFrame(data.groupby(['nationality'])['overall'].sum().reset_index())
count = pd.DataFrame(rating.groupby('nationality')['overall'].sum().reset_index())

trace = [go.Choropleth(
            colorscale = 'YlOrRd',
            locationmode = 'country names',
            locations = count['nationality'],
            text = count['nationality'],
            z = count['overall'],
)]

layout = go.Layout(title = 'Country vs Overall')

fig = go.Figure(data = trace, layout = layout)
py.iplot(fig)
# function to check details of a player using id

def playerdata(x):
    return data.loc[x,:]

x = playerdata(0)  #lionel messi, id = 0.
pd.set_option('display.max_rows', 200)
x = pd.DataFrame(x)
print(x)