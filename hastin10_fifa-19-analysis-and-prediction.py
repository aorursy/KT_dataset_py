# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import time 



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv')

print(df.shape)
df.head()
df.isnull().sum()
df.describe()
df.info()
def weight_correction(df):

    try:

        value = float(df[:-3])

    except:

        value = 0

    return value

df['Weight'] = df.Weight.apply(weight_correction)
df.Weight = pd.to_numeric(df.Weight)
df.Weight = df.Weight.replace(0, np.nan)
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



df['Value'] = df['Value'].apply(value_to_int)

df['Wage'] = df['Wage'].apply(value_to_int)



df.Value = df.Value.replace(0, np.nan)

df.Wage = df.Wage.replace(0, np.nan)
df.Weight.isna().sum()
df.Weight.mean()
df['Weight'].fillna(df.Weight.mean(), inplace = True)
df.Height.isna().sum()
plt.figure(figsize = (20, 10))

sns.countplot(x = 'Height', data = df)

plt.show()
df['Height'].fillna("5'11", inplace = True)
wf_missing = df['Weak Foot'].isna()

wf_missing.sum()
weak_foot_prob = df['Weak Foot'].value_counts(normalize = True)

weak_foot_prob
df.loc[wf_missing, 'Weak Foot'] = np.random.choice(weak_foot_prob.index, size = wf_missing.sum(), p = weak_foot_prob.values)
pf_missing = df['Preferred Foot'].isna()

pf_missing.sum()
foot_distribution = df['Preferred Foot'].value_counts(normalize = True)

foot_distribution
df.loc[pf_missing, 'Preferred Foot'] = np.random.choice(foot_distribution.index, size = pf_missing.sum(), p = foot_distribution.values)
fp_missing = df.Position.isna()

fp_missing.sum()
position_prob = df.Position.value_counts(normalize = True)

position_prob 
df.loc[fp_missing, 'Position'] = np.random.choice(position_prob.index, p = position_prob.values, size = fp_missing.sum())
fs_missing = df['Skill Moves'].isna()

fs_missing.sum()
skill_moves_prob = df['Skill Moves'].value_counts(normalize=True)

skill_moves_prob
df.loc[fs_missing, 'Skill Moves'] = np.random.choice(skill_moves_prob.index, p = skill_moves_prob.values, size = fs_missing.sum())
bt_missing = df['Body Type'].isna()

bt_missing.sum()
bt_prob = df['Body Type'].value_counts(normalize = True)

bt_prob
df.loc[bt_missing, 'Body Type'] = np.random.choice(['Normal', 'Lean'], p = [.63,.37], size = bt_missing.sum())
wage_missing = df['Wage'].isna()

wage_missing.sum()
wage_prob = df.Wage.value_counts(normalize = True)

wage_prob
df.loc[wage_missing, 'Wage'] = np.random.choice(wage_prob.index, p = wage_prob.values, size = wage_missing.sum())
wr_missing = df['Work Rate'].isna()

wr_missing.sum()
wr_prob = df['Work Rate'].value_counts(normalize=True)

wr_prob
df.loc[wr_missing, 'Work Rate'] = np.random.choice(wr_prob.index, p = wr_prob.values, size = wr_missing.sum())
ir_missing = df['International Reputation'].isna()

ir_missing.sum()
ir_prob = df['International Reputation'].value_counts(normalize = True)

ir_prob
df.loc[ir_missing, 'International Reputation'] = np.random.choice(ir_prob.index, p = ir_prob.values, size = ir_missing.sum())
# filling the missing value for the continous variables for proper data visualization



df['ShortPassing'].fillna(df['ShortPassing'].mean(), inplace = True)

df['Volleys'].fillna(df['Volleys'].mean(), inplace = True)

df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True)

df['Curve'].fillna(df['Curve'].mean(), inplace = True)

df['FKAccuracy'].fillna(df['FKAccuracy'].mean(), inplace = True)

df['LongPassing'].fillna(df['LongPassing'].mean(), inplace = True)

df['BallControl'].fillna(df['BallControl'].mean(), inplace = True)

df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean(), inplace = True)

df['Finishing'].fillna(df['Finishing'].mean(), inplace = True)

df['Crossing'].fillna(df['Crossing'].mean(), inplace = True)

df['Acceleration'].fillna(df['Acceleration'].mean(), inplace = True)

df['SprintSpeed'].fillna(df['SprintSpeed'].mean(), inplace = True)

df['Agility'].fillna(df['Agility'].mean(), inplace = True)

df['Reactions'].fillna(df['Reactions'].mean(), inplace = True)

df['Balance'].fillna(df['Balance'].mean(), inplace = True)

df['ShotPower'].fillna(df['ShotPower'].mean(), inplace = True)

df['Jumping'].fillna(df['Jumping'].mean(), inplace = True)

df['Stamina'].fillna(df['Stamina'].mean(), inplace = True)

df['Strength'].fillna(df['Strength'].mean(), inplace = True)

df['LongShots'].fillna(df['LongShots'].mean(), inplace = True)

df['Aggression'].fillna(df['Aggression'].mean(), inplace = True)

df['Interceptions'].fillna(df['Interceptions'].mean(), inplace = True)

df['Positioning'].fillna(df['Positioning'].mean(), inplace = True)

df['Vision'].fillna(df['Vision'].mean(), inplace = True)

df['Penalties'].fillna(df['Penalties'].mean(), inplace = True)

df['Composure'].fillna(df['Composure'].mean(), inplace = True)

df['Marking'].fillna(df['Marking'].mean(), inplace = True)

df['StandingTackle'].fillna(df['StandingTackle'].mean(), inplace = True)

df['SlidingTackle'].fillna(df['SlidingTackle'].mean(), inplace = True)
df['Loaned From'].fillna('None', inplace = True)

df['Club'].fillna('No Club', inplace = True)
df.fillna(0, inplace = True)
def defending(df):

    return int(round((df[['Marking', 'StandingTackle', 

                               'SlidingTackle']].mean()).mean()))



def general(df):

    return int(round((df[['HeadingAccuracy', 'Dribbling', 'Curve', 

                               'BallControl']].mean()).mean()))



def mental(df):

    return int(round((df[['Aggression', 'Interceptions', 'Positioning', 

                               'Vision','Composure']].mean()).mean()))



def passing(df):

    return int(round((df[['Crossing', 'ShortPassing', 

                               'LongPassing']].mean()).mean()))



def mobility(df):

    return int(round((df[['Acceleration', 'SprintSpeed', 

                               'Agility','Reactions']].mean()).mean()))

def power(df):

    return int(round((df[['Balance', 'Jumping', 'Stamina', 

                               'Strength']].mean()).mean()))



def rating(df):

    return int(round((df[['Potential', 'Overall']].mean()).mean()))



def shooting(df):

    return int(round((df[['Finishing', 'Volleys', 'FKAccuracy', 

                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))
df['Defending'] = df.apply(defending, axis = 1)

df['General'] = df.apply(general, axis = 1)

df['Mental'] = df.apply(mental, axis = 1)

df['Passing'] = df.apply(passing, axis = 1)

df['Mobility'] = df.apply(mobility, axis = 1)

df['Power'] = df.apply(power, axis = 1)

df['Rating'] = df.apply(rating, axis = 1)

df['Shooting'] = df.apply(shooting, axis = 1)
players = df[['Name', 'Defending', 'General', 'Mental', 'Passing',

                'Mobility', 'Power', 'Rating', 'Shooting', 'Age',

                'Nationality', 'Club']]
plt.figure(figsize = (10, 10))

ax = sns.countplot(x = 'Skill Moves', data = df, palette = 'bright')

ax.set_title(label = 'Count of players on the basis of their skill moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Rating of skill moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
labels = ['3', '2', '4', '5', '1']

sizes = df['Weak Foot'].value_counts()

plt.rcParams['figure.figsize'] = (10, 10)

plt.pie(sizes, labels = labels)

plt.title('Distribution of players on the basis of their weak foot rating', fontsize = 20)

plt.legend()

plt.show()
plt.figure(figsize = (10, 10))

ax = sns.countplot(x = 'Preferred Foot', data = df, palette = 'deep')

ax.set_title(label = 'Count of players on the basis of their preferred foot', fontsize = 20)

ax.set_xlabel(xlabel = 'Preferred foot', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
labels = ['1', '2', '3', '4', '5']

sizes = df['International Reputation'].value_counts()

explode = [0.1, 0.2, 0.3, 0.7, 0.9]



plt.rcParams['figure.figsize'] = (10, 10)

plt.pie(sizes, labels = labels, explode = explode)

plt.title('Distribution of the international reputation of the players', fontsize = 20)

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (10, 10)

sns.distplot(df['Wage'], color = 'blue')

plt.xlabel('Wage Range for Players', fontsize = 16)

plt.ylabel('Count of the Players', fontsize = 16)

plt.title('Distribution of Wages of Players', fontsize = 20)

plt.show()
plt.figure(figsize = (20, 10))

plt.style.use('fivethirtyeight')

ax = sns.countplot('Position', data = df, palette = 'Reds_r')

ax.set_xlabel(xlabel = 'Different positions in football', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

ax.set_title(label = 'Count of players on the basis of position', fontsize = 20)

plt.show()
plt.figure(figsize = (20, 10))

plt.style.use('fast')

sns.countplot(x = 'Work Rate', data = df, palette = 'hls')

plt.title('Count of players on the basis of work rate', fontsize = 20)

plt.xlabel('Work rates', fontsize = 16)

plt.ylabel('Count', fontsize = 16)

plt.show()
x = df.Special

plt.figure(figsize = (20, 10))

plt.style.use('tableau-colorblind10')

ax = sns.distplot(x, bins = 50, kde = True, color = 'g')

ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count',fontsize = 16)

ax.set_title(label = 'Count of players on the basis of their speciality score', fontsize = 20)

plt.show()
x = df.Potential

plt.figure(figsize = (20, 10))

plt.style.use('seaborn-paper')

ax = sns.distplot(x, bins = 50, color = 'y')

ax.set_xlabel(xlabel = "Player\'s potential scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

ax.set_title(label = 'Count of players on the basis of potential scores', fontsize = 20)

plt.show()
sns.set(style = "dark", palette = "deep", color_codes = True)

x = df.Overall

plt.figure(figsize = (20, 10))

plt.style.use('ggplot')

ax = sns.distplot(x, bins = 50, color = 'r')

ax.set_xlabel(xlabel = "Player\'s Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

ax.set_title(label = 'Count of players on the basis of their overall scores', fontsize = 20)

plt.show()
plt.style.use('fast')

sns.jointplot(x = 'Age', y = 'Potential', data = df)

plt.show()
sns.jointplot(x = 'Special', y = 'Overall', data = df, joint_kws={'color':'orange'}, marginal_kws={'color':'blue'})

plt.show()
plt.style.use('dark_background')

df['Nationality'].value_counts().plot.bar(color = 'orange', figsize = (30, 15))

plt.title('Different Nations Participating in FIFA 2019', fontsize = 20)

plt.xlabel('Name of The Country', fontsize = 16)

plt.ylabel('Count', fontsize = 16)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 15)

plt.show()
player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties', 

    'ShortPassing', 'Volleys', 'Dribbling', 'Curve',

    'Finishing', 'Crossing', 'SprintSpeed', 'Reactions',

    'ShotPower', 'Stamina', 'Strength',

    'Positioning', 'StandingTackle', 'SlidingTackle'

)



from math import pi

idx = 1

plt.style.use('seaborn-bright')

plt.figure(figsize = (25,45))

for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(4))

    

    # number of variable

    categories = top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(9, 3, idx, polar = True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color = 'white', size = 10)

    

    # Draw ylabels

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color = "white", size = 9)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    # Plot data

    ax.plot(angles, values, linewidth = 1, linestyle = 'solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha = 0.1)

    

    plt.title(position_name, size = 10, y = 1.1)

    

    idx += 1
labels = np.array(['Acceleration', 'Strength', 'Finishing',  

    'LongPassing', 'Penalties', 

    'ShortPassing', 'Volleys',

    'Finishing', 'Crossing', 'SprintSpeed',

    'ShotPower'])



stats = df.loc[0, labels].values

stats_1 = df.loc[1, labels].values

stats_2 = df.loc[2, labels].values
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint = False)



# close the plot

stats = np.concatenate((stats, [stats[0]]))

stats_1 = np.concatenate((stats_1, [stats_1[0]]))

stats_2 = np.concatenate((stats_2, [stats_2[0]]))



angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure()

ax = fig.add_subplot(111, polar = True)

ax.plot(angles, stats, 'o-', linewidth = 2)

ax.fill(angles, stats, alpha = 0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title([df.loc[0, 'Name']], position = (0, 1))# Changing the 1st argument in df.loc will give the radar charts for different players

ax.grid(True)
fig = plt.figure()

ax_1 = fig.add_subplot(111, polar = True)

ax_1.plot(angles, stats_1, 'o-', linewidth = 2)

ax_1.fill(angles, stats_1, alpha = 0.25)

ax_1.set_thetagrids(angles * 180/np.pi, labels)

ax_1.set_title([df.loc[1, 'Name']], position = (0, 1))

ax_1.grid(True)
fig = plt.figure()

ax_2 = fig.add_subplot(111, polar = True)

ax_2.plot(angles, stats_2, 'o-', linewidth = 2)

ax_2.fill(angles, stats_2, alpha = 0.25)

ax_2.set_thetagrids(angles * 180/np.pi, labels)

ax_2.set_title([df.loc[2, 'Name']], position = (0, 1))

ax_2.grid(True)
plt.figure(figsize = (10, 10))

plt.style.use('seaborn-darkgrid')

plt.scatter(df['Overall'], df['International Reputation'])

plt.xlabel('Overall Ratings', fontsize = 16)

plt.ylabel('International Reputation', fontsize = 16)

plt.title('Ratings vs Reputation', fontsize = 20)

plt.show()
df.loc[df.groupby(df['Position'])['Potential'].idxmax()][['Name', 'Position', 'Overall', 'Potential', 'Age', 'Nationality', 'Club']]
df.loc[df.groupby(df['Position'])['Overall'].idxmax()][['Name', 'Position', 'Overall', 'Age', 'Nationality', 'Club']]
df.groupby('Age')['Overall'].mean().plot(figsize = (20, 10))

plt.xlabel('Age', fontsize = 16)

plt.ylabel('Mean', fontsize = 16)

plt.title('Mean of overall, age-wise', fontsize = 20)

plt.show()
df['Age'].value_counts()
plt.style.use('seaborn-deep')

plt.figure(figsize = (20, 10))

sns.countplot(x = 'Age', data = df)

plt.xlabel('Age', fontsize = 16)

plt.ylabel('Count', fontsize = 16)

plt.title('Count of players, age-wise', fontsize = 20)

plt.show()
df[df['Age'] > 40][['Name', 'Overall', 'Age', 'Nationality']]
df.sort_values('Age', ascending = True)[['Name', 'Age', 'Club', 'Nationality']].head(15)
df.sort_values('Age', ascending = False)[['Name', 'Age', 'Club', 'Nationality']].head(15)
df[df['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)
df[df['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
d = {'Overall': 'Average_Rating'}

best_overall_club_df = df.groupby('Club').agg({'Overall' : 'mean'}).rename(columns = d)

clubs = best_overall_club_df.Average_Rating.nlargest(5).index

print(clubs)
attck_list = ['Shooting', 'Power', 'Passing']



best_attack_df = players.groupby('Club')[attck_list].sum().sum(axis = 1)

clubs = best_attack_df.nlargest(6).index

print(clubs)
best_defense_df = players.groupby('Club')['Defending'].sum()

clubs = best_defense_df.nlargest(6).index

print(clubs)
df['Club'].value_counts().head(15)
some_clubs = ('Manchester United', 'Arsenal', 'Juventus', 'Paris Saint-Germain', 'Napoli', 'Manchester City',

             'Tottenham Hotspur', 'FC Barcelona', 'Inter', 'Chelsea', 'Real Madrid', 'Borussia Dortmund', 'Liverpool', 'Roma', 'Ajax')
df_clubs = df.loc[df['Club'].isin(some_clubs) & df['Overall']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = df_clubs['Club'], y = df_clubs['Overall'], palette = 'rocket')

ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)

ax.set_ylabel(ylabel = 'Overall score', fontsize = 16)

ax.set_title(label = 'Distribution of overall score in different popular clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
df_club = df.loc[df['Club'].isin(some_clubs) & df['Age']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = 'Club', y = 'Age', data = df_club, palette = 'magma')

ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 16)

ax.set_title(label = 'Distribution of ages in some popular clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
df_club = df.loc[df['Club'].isin(some_clubs) & df['Wage']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = 'Club', y = 'Wage', data = df_club, palette = 'Reds')

ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 16)

ax.set_title(label = 'Disstribution of wages in some popular clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
df_club = df.loc[df['Club'].isin(some_clubs) & df['International Reputation']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = 'Club', y = 'International Reputation', data = df_club, palette = 'bright')

ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)

ax.set_ylabel(ylabel = 'International reputation', fontsize = 16)

ax.set_title(label = 'Distribution of international reputation in some popular clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
df_clubs = df.loc[df['Club'].isin(some_clubs) & df['Weight']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = 'Club', y = 'Weight', data = df_clubs, palette = 'rainbow')

ax.set_xlabel(xlabel = 'Some popular clubs', fontsize = 16)

ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 16)

ax.set_title(label = 'Distribution of weight in different popular clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
df.groupby(df['Club'])['Nationality'].nunique().sort_values(ascending = False).head(11)
df.groupby(df['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10)
d = {'Overall': 'Average_Rating'}

best_overall_country_df = df.groupby('Nationality').agg({'Overall':'mean'}).rename(columns = d)

nations = best_overall_country_df.Average_Rating.nlargest(5).index

print(nations)
best_attack_nation_df = players.groupby('Nationality')[attck_list].sum().mean(axis = 1)

nations = best_attack_nation_df.nlargest(5).index

print(nations)
best_defense_nation_df = players.groupby('Nationality')['Defending'].sum()

nations = best_defense_nation_df.nlargest(5).index

print(nations)
df['Nationality'].value_counts().head(15)
some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Colombia', 'Japan', 'Netherlands')
df_countries = df.loc[df['Nationality'].isin(some_countries) & df['Weight']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['Weight'], palette = 'cubehelix')

ax.set_xlabel(xlabel = 'Countries', fontsize = 16)

ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 16)

ax.set_title(label = 'Distribution of weight of players from different countries', fontsize = 20)

plt.show()
df_countries = df.loc[df['Nationality'].isin(some_countries) & df['Overall']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['Overall'], palette = 'spring')

ax.set_xlabel(xlabel = 'Countries', fontsize = 16)

ax.set_ylabel(ylabel = 'Overall scores', fontsize = 16)

ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)

plt.show()
df_countries = df.loc[df['Nationality'].isin(some_countries) & df['Wage']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['Wage'], palette = 'hot')

ax.set_xlabel(xlabel = 'Countries', fontsize = 16)

ax.set_ylabel(ylabel = 'Wage', fontsize = 16)

ax.set_title(label = 'Distribution of wages of players from different countries', fontsize = 20)

plt.show()
df_countries = df.loc[df['Nationality'].isin(some_countries) & df['International Reputation']]

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x = df_countries['Nationality'], y = df_countries['International Reputation'], palette = 'autumn')

ax.set_xlabel(xlabel = 'Countries', fontsize = 16)

ax.set_ylabel(ylabel = 'International reputation', fontsize = 16)

ax.set_title(label = 'Distribution of international repuatation of players from different countries', fontsize = 20)

plt.show()
selected_columns = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']



df_selected = pd.DataFrame(df, columns = selected_columns)

df_selected.columns
df_selected.sample(5)
plt.rcParams['figure.figsize'] = (30, 30)

sns.heatmap(df_selected[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']].corr(), annot = True)



plt.title('Heatmap of the dataset', fontsize = 30)

plt.show()
GK_attributes = df[['GKPositioning','GKKicking','GKHandling','GKReflexes','GKDiving']]



plt.rcParams['figure.figsize'] = (10, 10)

sns.heatmap(GK_attributes.corr(), annot = True)



plt.title('correlations between attributes of goalkeeper', fontsize = 30)

plt.show()
dummy_df = df.copy()
print(dummy_df.keys())

print(dummy_df.shape)

print(dummy_df.select_dtypes(['O']).shape)        

print(dummy_df.select_dtypes([np.number]).shape)
dummy_df.drop(['GKPositioning', 'GKKicking', 'GKHandling', 'GKReflexes'], inplace = True, axis = 1)
dummy_df.drop(['Rating'], inplace = True, axis = 1)
dummy_df.drop(['StandingTackle', 'SlidingTackle'], inplace = True, axis = 1)
dummy_df.drop(['Interceptions'], inplace = True, axis = 1)
dummy_df.drop(['BallControl'], inplace = True, axis = 1)
dummy_df.drop(['LongShots'], inplace = True, axis = 1)
string_columns = dummy_df.select_dtypes(['O']).columns           

string_columns
dummy_df.drop(['Photo', 'Flag', 'Club Logo', 'Real Face', 'Joined', 'Loaned From', 'Contract Valid Until', 'Height', 'Release Clause'],inplace = True,axis = 1)
dummy_df.drop(['Preferred Foot', 'Nationality'], inplace = True, axis = 1)
dummy_df[['w_r_attack','w_r_defence']] = dummy_df['Work Rate'].str.split('/',expand=True)
dummy_df.w_r_attack = dummy_df.w_r_attack.str.strip()

dummy_df.w_r_defence = dummy_df.w_r_defence.str.strip()
dummy_df.w_r_defence = dummy_df.w_r_defence.map({'High':3,'Medium':2,'Low':1})

dummy_df.w_r_attack = dummy_df.w_r_attack.map({'High':3,'Medium':2,'Low':1})
dummy_df.drop(['Work Rate'],inplace = True,axis = 1)
dummy_df.drop(['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'], inplace = True, axis = 1)
dummy_df.drop(['Body Type'], inplace = True, axis = 1)
string_columns = dummy_df.select_dtypes(['O']).columns           

string_columns
le = LabelEncoder()

for column in string_columns[1:]:                                    

    dummy_df[column] = le.fit_transform(dummy_df[column])
string_columns = dummy_df.select_dtypes(['O']).columns      

string_columns
number_columns = dummy_df.select_dtypes([np.number]).columns           

number_columns
dummy_df.drop(['Unnamed: 0', 'Jersey Number', 'ID'], inplace = True, axis = 1)
dummy_df
from sklearn.preprocessing import MinMaxScaler
dummy_df_scaled = dummy_df.copy()
output_2 = dummy_df['Overall']

dummy_df.drop(['Overall'], inplace = True, axis = 1)
scaling = MinMaxScaler(copy = False).fit(dummy_df_scaled.iloc[:, 1:])

dummy_df_scaled.iloc[:, 1:] = scaling.transform(dummy_df_scaled.iloc[:, 1:])
dummy_df_scaled
from sklearn.model_selection import train_test_split,cross_val_score,KFold

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.metrics import mean_squared_error

from IPython.display import display
def model_performance(model, data, output_df):

    

    # Splitting the data into train and test set

    x_train, x_test, y_train, y_test = train_test_split(data, output_df)

    

    train_names = x_train.iloc[:,0]

    x_train = x_train.iloc[:,1:]

    test_names = x_test.iloc[:,0]

    x_test = x_test.iloc[:,1:]

    

    start = time.time()

    model.fit(x_train, y_train)

    print("fitting time : {}".format(time.time()-start))



    start = time.time()

    y_pred = model.predict(x_test)

    print("\nModel's score is :", model.score(x_test, y_test))          # Returns the coefficient of determination R^2 of the prediction.

    print("testing time : {}".format(time.time() - start))

    

    print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))

    

    #Cross-validation score

    crossScore = cross_val_score(model, X = data.iloc[:,1:], y = output_df, cv = KFold(n_splits = 5, shuffle = True)).mean()

    print("\nThe cross-validation score is:", crossScore) 

    

    comparisionRDF = pd.DataFrame(y_test)

    comparisionRDF['predicted'] = y_pred

    comparisionRDF['player Name'] = test_names

    comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])

    print("\nThe error in the prediction for each player is:")

    print(comparisionRDF)

    

    #Limits of error

    print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))

    

    model.get_params()              

    weights = model.coef_           # Coefficients of all features/columns

    bias = model.intercept_         # Constant term in a linear line equation

    print("\nweights are:",weights)

    print("Constant is :",bias)

    

    print("\nThe importance of each feature for the model is:")

    perm = PermutationImportance(model).fit(x_test, y_test)

    display(eli5.show_weights(perm, feature_names = x_test.columns.tolist()))

    

    # Visualising the results

    plt.figure(figsize=(20, 10))

    sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})

    plt.xlabel('Predictions')

    plt.ylabel('Overall')

    plt.title("Prediction of overall rating")

    plt.show()

    

    # Visualising the residual plot

    plt.figure(figsize=(20, 10))

    sns.scatterplot(y_pred, y_test - y_pred)

    plt.xlabel('Predictions')

    plt.ylabel('Residual')

    plt.title("Residual plot")

    plt.show()
output = dummy_df_scaled['Overall']

dummy_df_scaled.drop(['Overall'], inplace = True, axis = 1)
from sklearn.linear_model import LinearRegression
model = LinearRegression() 

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
from sklearn.linear_model import Ridge
model = Ridge()

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = Ridge(alpha = 0.05)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = Ridge(alpha = 0.5)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = Ridge(alpha = 5)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
from sklearn.linear_model import Lasso
model = Lasso(alpha = 0.05)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = Lasso(alpha = 0.05, max_iter = 10000)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = Lasso(alpha = 0.5, max_iter = 10000)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = Lasso(alpha = 5, max_iter = 10000)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha = 1, l1_ratio = 0.5)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = ElasticNet(alpha = 1, l1_ratio = 0.3)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
model = ElasticNet(alpha = 1, l1_ratio = 0.7)

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
from sklearn.svm import LinearSVR
model = LinearSVR()

model_performance(model, dummy_df_scaled, output)

model_performance(model, dummy_df, output_2)
from sklearn.model_selection import RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(dummy_df_scaled, output)

    

train_names = x_train.iloc[:,0]

x_train = x_train.iloc[:,1:]

test_names = x_test.iloc[:,0]

x_test = x_test.iloc[:,1:]
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

print('Parameters currently in use:\n')

print(model.get_params())
# Create the random grid

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees in random forest

               'max_features': ['auto', 'sqrt'],# Number of features to consider at every split

               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in tree

               'min_samples_split': [2, 5, 10, 20],# Minimum number of samples required to split a node

               'min_samples_leaf': [1, 2, 4, 10, 25],# Minimum number of samples required at each leaf node 

               'bootstrap': [True, False]}# Method of selecting samples for training each tree

print(random_grid)

        

# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 5 different combinations

model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 2, cv = 3, verbose = 1, n_jobs = 1)

        

# Fit the random search model

model_random.fit(x_train, y_train)



print("\nThe best parameters for the model:", model_random.best_params_)

        

model = model_random.best_estimator_

        

start = time.time()

model.fit(x_train, y_train)

print("fitting time : {}".format(time.time()-start))

        

start = time.time()

y_pred = model.predict(x_test)

print("testing time : {}".format(time.time() - start))



print("\nThe importance of each feature for the model:", model.feature_importances_)
print("\nModel's score is :", model.score(x_test, y_test))

print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))

        

crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()

print("\nThe cross-validation score is:", crossScore)

        

comparisionRDF = pd.DataFrame(y_test)

comparisionRDF['predicted'] = y_pred

comparisionRDF['player Name'] = test_names

comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])

print("\nThe error in the prediction for each player is:")

print(comparisionRDF)

    

#Limits of error

print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))



print('\nParameters currently in use:\n')

print(model.get_params())

    

#Visualising the results

plt.figure(figsize=(20, 10))

sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Prediction of overall rating")

plt.show()



#Visualising the residual plot

plt.figure(figsize=(20, 10))

sns.scatterplot(y_pred, y_test - y_pred)

plt.xlabel('Predictions')

plt.ylabel('Residual')

plt.title("Residual plot")

plt.show()
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()

print('Parameters currently in use:\n')

print(model.get_params())
# Create the random grid

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees in random forest

               'max_features': ['auto', 'sqrt'],# Number of features to consider at every split

               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in tree

               'min_samples_split': [2, 5, 10, 20],# Minimum number of samples required to split a node

               'min_samples_leaf': [1, 2, 4, 10, 25],# Minimum number of samples required at each leaf node 

               'bootstrap': [True, False]}# Method of selecting samples for training each tree

print(random_grid)

        

# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 10 different combinations

model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 2, cv = 3, verbose = 1, n_jobs = 1)

        

# Fit the random search model

model_random.fit(x_train, y_train)





print("\nThe best parameters for the model:", model_random.best_params_)

        

model = model_random.best_estimator_

        

start = time.time()

model.fit(x_train, y_train)

print("fitting time : {}".format(time.time()-start))

        

start = time.time()

y_pred = model.predict(x_test)

print("testing time : {}".format(time.time() - start))



print("\nThe importance of each feature for the model:", model.feature_importances_)
print("\nModel's score is :", model.score(x_test, y_test))

print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))

        

crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()

print("\nThe cross-validation score is:", crossScore)

        

comparisionRDF = pd.DataFrame(y_test)

comparisionRDF['predicted'] = y_pred

comparisionRDF['player Name'] = test_names

comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])

print("\nThe error in the prediction for each player is:")

print(comparisionRDF)

    

#Limits of error

print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))



print('\nParameters currently in use:\n')

print(model.get_params())

    

#Visualising the results

plt.figure(figsize=(20, 10))

sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Prediction of overall rating")

plt.show()



#Visualising the residual plot

plt.figure(figsize=(20, 10))

sns.scatterplot(y_pred, y_test - y_pred)

plt.xlabel('Predictions')

plt.ylabel('Residual')

plt.title("Residual plot")

plt.show()
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()

print('Parameters currently in use:\n')

print(model.get_params())
# Create the random grid

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)],# Maximum number of trees at which boosting is terminated

               'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],# Learning rate of the algorithm

               'loss': ['linear', 'square', 'exponential']}# Loss function to be used for updating the weights

print(random_grid)

        

# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)



# Fit the random search model

model_random.fit(x_train, y_train)

        

print("\nThe best parameters for the model:", model_random.best_params_)

        

model = model_random.best_estimator_

        

start = time.time()

model.fit(x_train, y_train)

print("fitting time : {}".format(time.time()-start))

        

start = time.time()

y_pred = model.predict(x_test)

print("testing time : {}".format(time.time() - start))        

        

print("\nThe weights given to different estimators:", model.estimator_weights_)

        

print("\nThe errors of different estimators:", model.estimator_errors_)
print("\nModel's score is :", model.score(x_test, y_test))

print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))

        

crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()

print("\nThe cross-validation score is:", crossScore)

        

comparisionRDF = pd.DataFrame(y_test)

comparisionRDF['predicted'] = y_pred

comparisionRDF['player Name'] = test_names

comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])

print("\nThe error in the prediction for each player is:")

print(comparisionRDF)

    

#Limits of error

print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))



print('\nParameters currently in use:\n')

print(model.get_params())

    

#Visualising the results

plt.figure(figsize=(20, 10))

sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Prediction of overall rating")

plt.show()



#Visualising the residual plot

plt.figure(figsize=(20, 10))

sns.scatterplot(y_pred, y_test - y_pred)

plt.xlabel('Predictions')

plt.ylabel('Residual')

plt.title("Residual plot")

plt.show()
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()

print('Parameters currently in use:\n')

print(model.get_params())
# Create the random grid

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees 

               'max_features': ['auto', 'sqrt'],# Number of features to consider at every split

               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in tree

               'min_samples_split': [2, 5, 10, 20],# Minimum number of samples required to split a node

               'min_samples_leaf': [1, 2, 4, 10, 25]}# Minimum number of samples required at each leaf node

print(random_grid)

        

# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)



# Fit the random search model

model_random.fit(x_train, y_train)

        

print("\nThe best parameters for the model:", model_random.best_params_)

        

model = model_random.best_estimator_

        

start = time.time()

model.fit(x_train, y_train)

print("fitting time : {}".format(time.time()-start))

        

start = time.time()

y_pred = model.predict(x_test)

print("testing time : {}".format(time.time() - start))
print("\nModel's score is :", model.score(x_test, y_test))

print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))

        

crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()

print("\nThe cross-validation score is:", crossScore)

        

comparisionRDF = pd.DataFrame(y_test)

comparisionRDF['predicted'] = y_pred

comparisionRDF['player Name'] = test_names

comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])

print("\nThe error in the prediction for each player is:")

print(comparisionRDF)

    

#Limits of error

print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))



print('\nParameters currently in use:\n')

print(model.get_params())

    

#Visualising the results

plt.figure(figsize=(20, 10))

sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Prediction of overall rating")

plt.show()



#Visualising the residual plot

plt.figure(figsize=(20, 10))

sns.scatterplot(y_pred, y_test - y_pred)

plt.xlabel('Predictions')

plt.ylabel('Residual')

plt.title("Residual plot")

plt.show()
import xgboost as xgb
model = xgb.XGBRegressor()

print('Parameters currently in use:\n')

print(model.get_params())
# Create the random grid

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],# Number of trees

               'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],# Learning rate of the algorithm

               'min_child_weight': [1, 3, 5, 7, 9],# Minimum sum of instance weight needed in a child

               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],# Maximum number of levels in a tree

               'colsample_bytree': [0.5, 0.8, 1],# Subsample ratio of columns when constructing each tree,

               'scale_pos_weight': [1, 2, 3, 4, 5]}# Balancing of positive and negative weights

print(random_grid)

        

# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)



# Fit the random search model

model_random.fit(x_train, y_train)

        

print("\nThe best parameters for the model:", model_random.best_params_)

        

model = model_random.best_estimator_

        

start = time.time()

model.fit(x_train, y_train)

print("fitting time : {}".format(time.time()-start))

        

start = time.time()

y_pred = model.predict(x_test)

print("testing time : {}".format(time.time() - start))

        

print("\nThe underlying xgboost booster of this model:", model.get_booster())

        

print("\nThe number of xgboost boosting rounds:", model.get_num_boosting_rounds())

    

print("\nXgboost type parameters:", model.get_xgb_params())        
print("\nModel's score is :", model.score(x_test, y_test))

print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))

        

crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()

print("\nThe cross-validation score is:", crossScore)

        

comparisionRDF = pd.DataFrame(y_test)

comparisionRDF['predicted'] = y_pred

comparisionRDF['player Name'] = test_names

comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])

print("\nThe error in the prediction for each player is:")

print(comparisionRDF)

    

#Limits of error

print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))



print('\nParameters currently in use:\n')

print(model.get_params())

    

#Visualising the results

plt.figure(figsize=(20, 10))

sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Prediction of overall rating")

plt.show()



#Visualising the residual plot

plt.figure(figsize=(20, 10))

sns.scatterplot(y_pred, y_test - y_pred)

plt.xlabel('Predictions')

plt.ylabel('Residual')

plt.title("Residual plot")

plt.show()
from sklearn.neighbors import KNeighborsRegressor
# Create the random grid

random_grid = {'n_neighbors': [5, 10, 15, 20],# Number of neighbors

               'weights': ['uniform', 'distance'],# Whether to weigh each point in the neighborhood equally or by the inverse of their distance

               'leaf_size': [20, 30, 40, 50]}# To be passed to the algorithm which will be used to compute the nearest neighbors

print(random_grid)



# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 3, verbose = 1, n_jobs = 1)



# Fit the random search model

model_random.fit(x_train, y_train)

        

print("\nThe best parameters for the model:", model_random.best_params_)

        

model = model_random.best_estimator_

        

start = time.time()

model.fit(x_train, y_train)

print("fitting time : {}".format(time.time()-start))

        

start = time.time()

y_pred = model.predict(x_test)

print("testing time : {}".format(time.time() - start))
print("\nModel's score is :", model.score(x_test, y_test))

print('\nRMSE : '+str(np.sqrt(mean_squared_error(y_test, y_pred))))

        

crossScore = cross_val_score(model, X = dummy_df_scaled.iloc[:,1:], y = output, cv = KFold(n_splits = 5, shuffle = True)).mean()

print("\nThe cross-validation score is:", crossScore)

        

comparisionRDF = pd.DataFrame(y_test)

comparisionRDF['predicted'] = y_pred

comparisionRDF['player Name'] = test_names

comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Overall'])

print("\nThe error in the prediction for each player is:")

print(comparisionRDF)

    

#Limits of error

print("\nLimit of error of our model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))



print('\nParameters currently in use:\n')

print(model.get_params())

    

#Visualising the results

plt.figure(figsize=(20, 10))

sns.regplot(y_pred, y_test, scatter_kws = {'color':'lime'}, line_kws = {'color':'red'})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Prediction of overall rating")

plt.show()



#Visualising the residual plot

plt.figure(figsize=(20, 10))

sns.scatterplot(y_pred, y_test - y_pred)

plt.xlabel('Predictions')

plt.ylabel('Residual')

plt.title("Residual plot")

plt.show()
from keras.layers import Input,Dense

from keras.models import Model

from keras.optimizers import Adam
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(dummy_df_scaled, output)

    

train_names_nn = x_train_nn.iloc[:,0]

x_train_nn = x_train_nn.iloc[:,1:]

test_names_nn = x_test_nn.iloc[:,0]

x_test_nn = x_test_nn.iloc[:,1:]
input_layer = Input((dummy_df_scaled.shape[1] - 1,))

y = Dense(64, kernel_initializer = 'he_normal', activation = 'relu')(input_layer)

y = Dense(32, kernel_initializer = 'he_normal', activation = 'relu')(y)

y = Dense(8, kernel_initializer = 'he_normal', activation = 'relu')(y)

y = Dense(1, kernel_initializer = 'he_normal', activation = 'sigmoid')(y)



model = Model(inputs = input_layer, outputs = y)

model.compile(optimizer = Adam(lr = 0.001), loss = 'mse', metrics = ['mean_squared_error'])

model.summary()
history = model.fit(x_train_nn, y_train_nn, epochs = 1000, batch_size = 512)
plt.plot(history.history['loss'], label = 'train')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend()

plt.show()
scores = model.evaluate(x_test,y_test)

print("Test Set MSE(loss, metric):",scores)