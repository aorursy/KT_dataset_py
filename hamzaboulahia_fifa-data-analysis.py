# Essential packages for analysis



import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from matplotlib import gridspec

from IPython.display import display, HTML

from math import pi

plt.style.use('fivethirtyeight')



%matplotlib inline



# ML packages

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn import linear_model, tree, svm

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier



# Input files path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import math
# Reading the .csv files from the input path



df15 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_15.csv')

df16 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_16.csv')

df17 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_17.csv')

df18 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_18.csv')

df19 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_19.csv')

df20 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')

display(df15.head(3))

display(df16.head(3))

display(df17.head(3))

display(df18.head(3))

display(df19.head(3))

display(df20.head(3))
# Checking the shape of each dataframe



l = df15.shape,df16.shape,df17.shape,df18.shape,df19.shape,df20.shape

shape = pd.DataFrame(l)

shape.index = [2015,2016,2017,2018,2019,2020]

shape.columns = ['rows','columns']

shape
# Column names list



print(list(df15.columns))
# Checking the NULL values in FIFA 20 dataframe



temp = df15.isna().sum().reset_index()

temp.columns = ['columns','na_nbr']

temp.query('na_nbr!=0')
# Checking head values of each column



display(df17.head())

display(df17[['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys']].head())

df17[['potential','value_eur','wage_eur','player_positions','preferred_foot','international_reputation','weak_foot',

      'skill_moves','work_rate','body_type','release_clause_eur','team_position','team_jersey_number','pace','shooting','passing','dribbling']].head()
display(df16['body_type'].value_counts())
df20.team_position.value_counts()
df15.player_traits.value_counts().index
l_df = [df15,df16,df17,df18,df19,df20];
# Filling missing players attributes with '0', and then evaluating these attributes to convert them from string format to float



# Creating lists for columns names:

# Columns to drop:

c_drop = ['player_url', 'real_face', 'player_tags', 'loaned_from', 'joined', 'release_clause_eur', 'contract_valid_until', 'nation_position', 'nation_jersey_number', 'gk_diving', 'gk_handling', 

          'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm',

          'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb']

# Different Attribute columns grouped in lists:

att_cols = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys']

skill_cols = ['skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control']

movements_cols = ['movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance']

power_cols = ['power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots']

mentality_cols = ['mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure']

defending_cols = ['defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle']

gk_cols = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']



attributes_cols = att_cols+skill_cols+movements_cols+power_cols+mentality_cols+defending_cols+gk_cols

l_attributes = [att_cols, skill_cols, movements_cols, power_cols, mentality_cols, defending_cols, gk_cols]



for k in l_df:

    for i in l_attributes:

        for j in i:

            k[j].fillna('0',inplace = True)





for k in l_df:

    for i in l_attributes:

        for j in i:

            temp = list(k[j])

            tempp=[]

            for item in temp:

                try:

                    item = eval(item)

                    tempp.append(item)

                except:

                    tempp.append(item)

            k[j] = tempp
# Checking that the values of form 'XX+X' are converted to int correctly:



df17[attributes_cols].head()
# Reduction of the number of features by averaging the features that belong to the same group in one representative feature.

# Creating new attributes columns



for item in l_df:

    item['attack_oa'] = round(item[att_cols].mean(axis=1, skipna = True))

    item['skill_oa'] = round(item[skill_cols].mean(axis=1, skipna = True))

    item['movements_oa'] = round(item[movements_cols].mean(axis=1, skipna = True))

    item['power_oa'] = round(item[power_cols].mean(axis=1, skipna = True))

    item['mentality_oa'] = round(item[mentality_cols].mean(axis=1, skipna = True))

    item['defending_oa'] = round(item[defending_cols].mean(axis=1, skipna = True))

    item['gk_oa'] = round(item[gk_cols].mean(axis=1, skipna = True))



# Dropping old attributes columns

for item in l_df:

    item.drop(c_drop, axis=1, inplace=True)

    item.drop(att_cols, axis = 1, inplace = True)

    item.drop(skill_cols, axis = 1, inplace = True)

    item.drop(movements_cols, axis = 1, inplace = True)

    item.drop(power_cols, axis = 1, inplace = True)

    item.drop(mentality_cols, axis = 1, inplace = True)

    item.drop(defending_cols, axis = 1, inplace = True)

    item.drop(gk_cols, axis = 1, inplace = True)
# Converting the player traits column to a numerical variable so that we can use it for later analysis and estimations.

# Assigning a value to each player trait based on its significance and importance.





l15 = ','.join(list(df15.player_traits.value_counts().index))+','

l16 = ','.join(list(df16.player_traits.value_counts().index))+','

l17 = ','.join(list(df17.player_traits.value_counts().index))+','

l18 = ','.join(list(df18.player_traits.value_counts().index))+','

l19 = ','.join(list(df19.player_traits.value_counts().index))+','

l20 = ','.join(list(df20.player_traits.value_counts().index))

l = l15+l16+l17+l18+l19+l20

ul = l.split(',')

ul = [item.replace("(CPU AI Only)","").strip() for item in ul]

traits_set = sorted(list(set(ul)), key=str.lower)

traits_value = [3,1,2,3,2,2,2,3,2,4,3,2,1,3,5,5,4,3,2,1,4,1,3,2,4,1,5,3,4,4,3,2,3,5,5,1,3,4,5,3,2,3,3,4,2]

traits_dic = {}

for i,j in zip(traits_set,traits_value):

    traits_dic[i] = j

traits_dic
# Creating a new column for the trait coefficient



def calcul_trait_coef(s):

    coef=0

    try:

        s.strip()

        l = s.split(',')

        for i in l:

            coef = coef + traits_dic[i.replace("(CPU AI Only)","").strip()]

    except:

        return 0

    return coef



def create_traits_colum(df):

    col = []

    l = list(df.player_traits)

    for i in l:

        coef = calcul_trait_coef(i)

        col.append(coef)

    return col



# Creating new trait coefficient columns



df15['trait_coef'] = create_traits_colum(df15)

df16['trait_coef'] = create_traits_colum(df16)

df17['trait_coef'] = create_traits_colum(df17)

df18['trait_coef'] = create_traits_colum(df18)

df19['trait_coef'] = create_traits_colum(df19)

df20['trait_coef'] = create_traits_colum(df20)



# Dropping player traits column



for item in l_df:

    item.drop(['player_traits'], axis=1, inplace=True)
# Changing the NULL, RES, and SUB team positions values to the actual player position



for item in l_df:

    l_res = []

    l_sub = []

    l_null = []

    res_idx = list(item.query('team_position == "RES"').index)

    sub_idx = list(item.query('team_position == "SUB"').index)

    null_idx = list(item.team_position.isnull().index)

    for i in res_idx:

        temp = item.loc[i,'player_positions'].split(',')

        l_res.append(temp[0])

    for j in sub_idx:

        temp = item.loc[j,'player_positions'].split(',')

        l_sub.append(temp[0])

    for k in null_idx:

        temp = item.loc[k,'player_positions'].split(',')

        l_null.append(temp[0])

    item.loc[res_idx,'team_position'] = l_res

    item.loc[sub_idx,'team_position'] = l_sub

    item.loc[null_idx,'team_position'] = l_null
# Dropping the player positions column



for item in l_df:

    item.drop('player_positions', axis=1, inplace=True)
# Checking that each player has a team position



df20.team_position.value_counts()
# Checking for remaining NULL values



temp = df17.isna().sum().reset_index()

temp.columns = ['columns','na_nbr']

temp.query('na_nbr!=0')
# Setting every non-uniform player body type to Normal



for item in l_df[1:]:

    temp = item.body_type.value_counts()

    l = list(temp[3:].index)

    for i in l:

        idx = list(item[item['body_type']==i].index)

        item.loc[idx,'body_type']='Normal'

df20.body_type.value_counts()
# Changing GK NULL attributes to 0, and changing players NULL team jersey number to 0



for item in l_df:

    gk_idx = list(item.pace.isnull())

    item.loc[gk_idx,['pace','shooting','passing','dribbling','defending','physic']] = 0

    no_team_idx = list(item.team_jersey_number.isnull())

    item.loc[no_team_idx,'team_jersey_number'] = 0
# Checking for NULL variable after cleaning



temp = df17.isna().sum().reset_index()

temp.columns = ['columns','na_nbr']

temp.query('na_nbr!=0')
# Checking the final form of a dataframe



display(df17.head())

df17[['potential','value_eur','wage_eur','preferred_foot','international_reputation','weak_foot',

      'skill_moves','work_rate','body_type','team_position','team_jersey_number','pace','shooting','passing','dribbling']].head()
# Building and saving the master dataframe



years=['2015','2016','2017','2018','2019','2020']

k=0

for item in l_df:

    item['year'] = years[k]

    k=k+1



df = pd.concat(l_df)

df.to_csv('fifa_data_2015_to_2020.csv', index=False)
df = pd.read_csv('./fifa_data_2015_to_2020.csv')

df.head(7)
df.columns
def get_year(df,year):

    temp = df.query('year==@year').reset_index()

    return temp
df15 = get_year(df,2015)

df16 = get_year(df,2016)

df17 = get_year(df,2017)

df18 = get_year(df,2018)

df19 = get_year(df,2019)

df20 = get_year(df,2020)

l_df=[df15,df16,df17,df18,df19,df20]
plt.figure(figsize = (20,7))

sb.countplot(data = df, x = 'age', hue='year', palette = sb.cubehelix_palette(6, start=1, rot=0, dark=0.2, light=.8, reverse=False))

plt.title('Age distribution by year')

plt.legend(loc=1);
plt.figure(figsize = (20,7))

sb.countplot(data = df, x = 'height_cm', hue='year', palette = sb.cubehelix_palette(6, start=2, rot=0, dark=0.2, light=.8, reverse=False))

plt.title('Height distribution by year')

plt.legend(loc=1);
plt.figure(figsize = (20,7))

sb.countplot(data = df, x = 'weight_kg', hue='year', palette = sb.cubehelix_palette(6, start=3, rot=0, dark=0.2, light=.8, reverse=False))

plt.title('Weight distribution by year')

plt.legend(loc=1);
plt.figure(figsize = (20,7))

temp_df = df.groupby(['year','nationality']).count()['sofifa_id'].sort_values(ascending=False).iloc[:50].reset_index()

nl = ['England', 'Germany', 'Argentina', 'Spain', 'France']

top5 = temp_df.query('nationality in @nl')

sb.barplot(data = top5, x = 'nationality', y='sofifa_id' , hue='year', palette = sb.cubehelix_palette(6, start=4, rot=0, dark=0.2, light=.8, reverse=False))

plt.title('Top 5 countries by player numbers from 2015 to 2020')

plt.legend(loc=1)

plt.ylabel('count');
plt.figure(figsize = (20,7))

sb.countplot(data = df, x = 'overall', hue='year', palette = sb.cubehelix_palette(6, start=5, rot=0, dark=0.2, light=.8, reverse=False))

plt.title('Overall rating distribution by year')

plt.legend(loc=1);
l=[]

for i in l_df:

    temp = i.head(5)[['short_name','overall','year']]

    l.append(temp)

fig = plt.figure(figsize=(22,12))

plt.suptitle('Comparaison of the top 5 players from 2015 to 2020',fontsize=22)

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

plt.subplot(231)

sb.barplot(data=l[0], x='short_name', y='overall', palette='Reds_r')

plt.ylim(85, 100)

plt.xlabel('Player Name')

plt.title('Top 5 most rated players in FIFA 2015')

plt.subplot(232)

sb.barplot(data=l[1], x='short_name', y='overall', palette='Reds_r')

plt.ylim(85, 100)

plt.xlabel('Player Name')

plt.title('Top 5 most rated players in FIFA 2016')

plt.subplot(233)

sb.barplot(data=l[2], x='short_name', y='overall', palette='Reds_r')

plt.ylim(85, 100)

plt.xlabel('Player Name')

plt.title('Top 5 most rated players in FIFA 2017')

plt.subplot(234)

sb.barplot(data=l[3], x='short_name', y='overall', palette='Reds_r')

plt.ylim(85, 100)

plt.xlabel('Player Name')

plt.title('Top 5 most rated players in FIFA 2018')

plt.subplot(235)

sb.barplot(data=l[4], x='short_name', y='overall', palette='Reds_r')

plt.ylim(85, 100)

plt.xlabel('Player Name')

plt.title('Top 5 most rated players in FIFA 2019')

plt.subplot(236)

sb.barplot(data=l[5], x='short_name', y='overall', palette='Reds_r')

plt.ylim(85, 100)

plt.xlabel('Player Name')

plt.title('Top 5 most rated players in FIFA 2020');
avg_ovr = df.groupby('year')['overall'].mean().reset_index()

avg_value = df.groupby('year')['value_eur'].mean().reset_index().drop(0,axis=0)



plt.figure(figsize=(20,6))

plt.subplot(121)

sb.barplot(data=avg_ovr, x='year', y='overall', palette= sb.cubehelix_palette(6, start=3, rot=0, dark=0.2, light=.8, reverse=False))

plt.ylim(0, 100)

plt.xlabel('Year')

plt.title('Average rating by year')

plt.yticks(np.arange(0, 100+1, 10))

plt.subplot(122)

sb.barplot(data=avg_value, x='year', y='value_eur', palette= sb.cubehelix_palette(6, start=3.5, rot=0, dark=0.2, light=.8, reverse=False))

plt.xlabel('Year')

plt.ylabel('Value (€)')

plt.title('Average player value by year');
# Calculating the proportion of right foot players and left foot players in our dataset.



pref_foot = df.groupby(['year','preferred_foot'])['preferred_foot'].count().rename('count', inplace=True).reset_index()

pref_foot = pref_foot.pivot(index='year', columns='preferred_foot', values='count')

s = pref_foot['Left']+pref_foot['Right']

pref_foot['Left'] = pref_foot['Left']/s

pref_foot['Right'] = pref_foot['Right']/s
pref_foot.plot(kind='bar', stacked=True, figsize=(10,8), rot=0)

plt.title('Proportions of Left/Right foot for each FIFA')

plt.xlabel('Year')

plt.ylabel('Proportion')

plt.ylim(0,1.2);
plt.figure(figsize=(20,6))

plt.tight_layout()

plt.subplot(121)

sb.countplot(data=df, x='year', hue='international_reputation', palette= sb.cubehelix_palette(5, start=3.5, rot=0, dark=0.2, light=.8, reverse=False))

plt.legend(loc='upper right', bbox_to_anchor=(1.11, 1))

plt.title('Distribution of international reputation levels by year')

plt.subplot(122)

sb.countplot(data=df, x='year', hue='skill_moves', palette= sb.cubehelix_palette(5, start=1.5, rot=0, dark=0.2, light=.8, reverse=False))

plt.legend(loc='upper right', bbox_to_anchor=(1.11, 1))

plt.title('Distribution of skill move levels by year');
plt.figure(figsize=(20,20))

plt.suptitle('Ditribution of players values by year on a logarithmic scale', y=0.92, fontsize=20)

plt.subplot(321)

sb.distplot(df16.value_eur, kde=False, bins=100)

plt.title('Ditribution of players values in FIFA 16')

plt.xscale('log')

plt.xlabel('Value (€)')

plt.ylabel('Count')

plt.subplot(322)

sb.distplot(df17.value_eur, kde=False, bins=100)

plt.title('Ditribution of players values in FIFA 17')

plt.xscale('log')

plt.xlabel('Value (€)')

plt.ylabel('Count')

plt.subplot(323)

sb.distplot(df18.value_eur, kde=False, bins=100)

plt.title('Ditribution of players values in FIFA 18')

plt.xscale('log')

plt.xlabel('Value (€)')

plt.ylabel('Count')

plt.subplot(324)

sb.distplot(df19.value_eur, kde=False, bins=100)

plt.title('Ditribution of players values in FIFA 19')

plt.xscale('log')

plt.xlabel('Value (€)')

plt.ylabel('Count')

plt.subplot(313)

sb.distplot(df20.value_eur, kde=False, bins=100)

plt.title('Ditribution of players values in FIFA 20')

plt.xscale('log')

plt.xlabel('Value (€)')

plt.ylabel('Count');
def get_players_noGK(df):

    idx_df_gk = list(df.query('pace==0').index)

    df_noGK = df.drop(idx_df_gk)

    return df_noGK
temp=get_players_noGK(df)

plt.figure(figsize=(15,20))

plt.suptitle('Box plots of the main non-GK players attributes by year', y=0.92, fontsize=20)

plt.subplot(321)

sb.boxplot(data=temp, x='year', y='pace',  linewidth=1.5)

plt.subplot(322)

sb.boxplot(data=temp, x='year', y='shooting',  linewidth=1.5)

plt.subplot(323)

sb.boxplot(data=temp, x='year', y='passing',  linewidth=1.5)

plt.subplot(324)

sb.boxplot(data=temp, x='year', y='dribbling',  linewidth=1.5)

plt.subplot(325)

sb.boxplot(data=temp, x='year', y='defending',  linewidth=1.5)

plt.subplot(326)

sb.boxplot(data=temp, x='year', y='physic',  linewidth=1.5);
temp=get_players_noGK(df)

plt.figure(figsize=(15,20))

plt.suptitle('Violin plots of main attributes overall rating  of non-GK players attributes by year', y=0.92, fontsize=20)

plt.subplot(321)

sb.violinplot(data=temp, x='year', y='attack_oa',  linewidth=1.5)

plt.ylabel('Attack Overall')

plt.subplot(322)

sb.violinplot(data=temp, x='year', y='skill_oa',  linewidth=1.5)

plt.ylabel('Skill Overall')

plt.subplot(323)

sb.violinplot(data=temp, x='year', y='movements_oa',  linewidth=1.5)

plt.ylabel('Movements Overall')

plt.subplot(324)

sb.violinplot(data=temp, x='year', y='mentality_oa',  linewidth=1.5)

plt.ylabel('Mentality Overall')

plt.subplot(313)

sb.violinplot(data=temp, x='year', y='defending_oa',  linewidth=1.5)

plt.ylabel('Defending Overall');
def get_best_by_pos(df):

    l=[]

    temp = df.groupby(['team_position'])[['overall']].max()

    for i in list(temp.index):

        ovr = temp.loc[i][0]

        best_i = df.query('team_position==@i & overall==@ovr').iloc[0]

        l.append(best_i)

    best_pos = pd.DataFrame(l)

    return best_pos
l=[]

l.append(get_best_by_pos(df15))

l.append(get_best_by_pos(df16))

l.append(get_best_by_pos(df17))

l.append(get_best_by_pos(df18))

l.append(get_best_by_pos(df19))

l.append(get_best_by_pos(df20))

best_pos = pd.concat(l, ignore_index=True)
plt.figure(figsize=(15,6))

plt.title('Best overall rating for each position by year')

sb.barplot(data=best_pos, x='team_position', y='overall', hue='year')

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

plt.xlabel('Team position')

plt.ylabel('Overall')

plt.ylim(60,100);
gb = best_pos.groupby('team_position')    

l = [gb.get_group(x) for x in gb.groups]



display(l[0].loc[:,['short_name','team_position','overall','club','nationality','year']])

display(l[1].loc[:,['short_name','team_position','overall','club','nationality','year']])

CSS = """

div.cell:nth-child(60) .output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(l[2].loc[:,['short_name','team_position','overall','club','nationality','year']])

display(l[3].loc[:,['short_name','team_position','overall','club','nationality','year']])

CSS = """

div.cell:nth-child(61) .output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(l[4].loc[:,['short_name','team_position','overall','club','nationality','year']])

display(l[5].loc[:,['short_name','team_position','overall','club','nationality','year']])

CSS = """

div.cell:nth-child(62) .output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(l[6].loc[:,['short_name','team_position','overall','club','nationality','year']])

display(l[7].loc[:,['short_name','team_position','overall','club','nationality','year']])

CSS = """

div.cell:nth-child(63) .output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(l[8].loc[:,['short_name','team_position','overall','club','nationality','year']])

display(l[9].loc[:,['short_name','team_position','overall','club','nationality','year']])

CSS = """

div.cell:nth-child(64) .output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(l[10].loc[:,['short_name','team_position','overall','club','nationality','year']])

display(l[11].loc[:,['short_name','team_position','overall','club','nationality','year']])

CSS = """

div.cell:nth-child(65) .output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(l[12].loc[:,['short_name','team_position','overall','club','nationality','year']])

display(l[13].loc[:,['short_name','team_position','overall','club','nationality','year']])

CSS = """

div.cell:nth-child(66) .output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(l[14].loc[:,['short_name','team_position','overall','club','nationality','year']])
interesting_cols = ['height_cm','weight_kg', 'nationality', 'club', 'overall', 'potential', 'value_eur',

       'wage_eur', 'preferred_foot','skill_moves', 'work_rate', 'body_type','pace', 'shooting', 'passing', 'dribbling',

       'defending', 'physic', 'attack_oa', 'skill_oa', 'movements_oa',

       'power_oa', 'mentality_oa', 'defending_oa', 'gk_oa', 'trait_coef']

select_df = df[interesting_cols]
plt.figure(figsize=(15,15))

sb.heatmap(select_df.drop(list(df.query('pace==0').index)).corr(), annot=True)

plt.title('Heatmap of correlation between some interesting variables for Non-GKs');
plt.figure(figsize=(10,10))

select_df_gk = select_df.drop(['skill_moves', 'work_rate', 'body_type','pace', 'shooting', 'passing', 'dribbling','defending', 'physic'], axis=1)

sb.heatmap(select_df_gk.drop(list(df.query('pace!=0').index)).corr(), annot=True)

plt.title('Heatmap of correlation between some interesting variables for GKs');
grid = sb.PairGrid(df[['overall', 'potential', 'value_eur','age']])

grid.map_diag(plt.hist)

grid.map_offdiag(plt.scatter);
df_temp = df.drop(list(df.query('pace==0').index))

grid = sb.PairGrid(df_temp[['pace', 'shooting', 'passing', 'dribbling','defending', 'physic']])

grid.map_diag(plt.hist)

grid.map_offdiag(plt.scatter);
grid = sb.PairGrid(df[['attack_oa', 'skill_oa', 'movements_oa','power_oa', 'mentality_oa', 'defending_oa', 'gk_oa']])

grid.map_diag(plt.hist)

grid.map_offdiag(plt.scatter);
# Some countries are passed as clubs so we need to clean that up to get the top clubs by ratings.



country_list = ['Argentina','Australia','Austria','Belgium','Bolivia','Brazil','Bulgaria','Cameroon','Canada','Chile','China','Colombia','Côte d’Ivoire'

,'Czech Republic','Denmark','Ecuador','Egypt','England','Finland','France','Germany','Greece','Hungary','Iceland','India'

,'Ireland','Italy','Mexico','Netherlands','New Zealand','Northern Ireland','Norway','Paraguay','Peru','Poland'

,'Portugal','Romania','Russia','Scotland','Slovenia','South Africa','Spain','Sweden','Switzerland','Turkey','United States','Uruguay','Venezuela','Wales']



def top_club_avg_ovr(df):

    temp = df.groupby(['club']).mean()[['overall','year']].sort_values(by='overall',ascending=False).reset_index()

    temp.drop(list(temp.query('club in @country_list').index),inplace=True)

    res = temp.head(3)

    return res
l=[]

for i in l_df:

    l.append(top_club_avg_ovr(i))

top_club = pd.concat(l)
years = [2015,2016,2017,2018,2019,2020]

barca = '#A50044'

bayern = '#DC052D'

paris = '#004170'

juve = '#000000'

real = "#FEBE10"

napoli = '#12A0D7'

chelsea = '#034694'

city = '#6CABDD'

liver = '#00B2A9'

united = '#DA291C'

roma = '#8E1F2F'

dortmund = '#FDE100'

leipzig= '#0C2043'





l=[]

for i in years:

    temp = top_club.query('year==@i')

    l.append(temp)

fig = plt.figure(figsize=(22,12))

plt.suptitle('Comparaison of the top 3 clubs by overall average from 2015 to 2020',fontsize=22)

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

plt.subplot(231)

sb.barplot(data=l[0], x='club', y='overall', palette=[barca,bayern,paris])

plt.ylim(60, 100)

plt.xlabel('Player Name')

plt.title('Top 3 clubs by overall average in FIFA 2015')

plt.subplot(232)

sb.barplot(data=l[1], x='club', y='overall', palette=[juve, bayern, barca])

plt.ylim(60, 100)

plt.xlabel('Player Name')

plt.title('Top 3 clubs by overall average in FIFA 2016')

plt.subplot(233)

sb.barplot(data=l[2], x='club', y='overall', palette=[juve,bayern,real])

plt.ylim(60, 100)

plt.xlabel('Player Name')

plt.title('Top 3 clubs by overall average in FIFA 2017')

plt.subplot(234)

sb.barplot(data=l[3], x='club', y='overall', palette=[barca,juve,real])

plt.ylim(60, 100)

plt.xlabel('Player Name')

plt.title('Top 3 clubs by overall average in FIFA 2018')

plt.subplot(235)

sb.barplot(data=l[4], x='club', y='overall', palette=[juve,barca,napoli])

plt.ylim(60, 100)

plt.xlabel('Player Name')

plt.title('Top 3 clubs by overall average in FIFA 2019')

plt.subplot(236)

sb.barplot(data=l[5], x='club', y='overall', palette=[bayern,real,juve])

plt.ylim(60, 100)

plt.xlabel('Player Name')

plt.title('Top 3 clubs by overall average in FIFA 2020');
LL=["FC Barcelona","Real Madrid"]

PL=["Chelsea","Manchester United", "Liverpool", "Manchester City"]

SA=["Juventus","Napoli","Roma"]

BL=["FC Bayern München","Borussia Dortmund","RB Leipzig"]



LLdf = df.query('club in @LL').groupby(['year','club']).mean()['overall'].reset_index()

PLdf = df.query('club in @PL').groupby(['year','club']).mean()['overall'].reset_index()

SAdf = df.query('club in @SA').groupby(['year','club']).mean()['overall'].reset_index()

BLdf = df.query('club in @BL').groupby(['year','club']).mean()['overall'].reset_index()



plt.figure(figsize=(20,12))

plt.suptitle('Evolution of some of the best clubs in different leagues in terms of average overall rating', y=0.95,fontsize='20')

plt.subplot(221)

sb.lineplot(data=LLdf, x='year', y='overall', hue='club', palette=[barca,real], marker='o', markersize=10 , legend=False)

plt.legend(LL, title='Clubs')

plt.title('La Liga')

plt.subplot(222)

sb.lineplot(data=PLdf, x='year', y='overall', hue='club', palette=[chelsea, liver, city, united], marker='o', markersize=10, legend=False)

plt.legend(PL, title='Clubs')

plt.title('Premier League')

plt.subplot(223)

sb.lineplot(data=SAdf, x='year', y='overall', hue='club', palette=[juve, napoli, roma], marker='o', markersize=10, legend=False)

plt.legend(SA, title='Clubs')

plt.title('Serie A')

plt.subplot(224)

sb.lineplot(data=BLdf, x='year', y='overall', hue='club', palette=[dortmund, bayern, leipzig], marker='o', markersize=10, legend=False)

plt.title('Bundesliga')

plt.legend(BL, title='Clubs');
crm_df = df.query('short_name == "L. Messi" | short_name == "Cristiano Ronaldo"')

plt.figure(figsize=(8,5))

ax = sb.barplot(data=crm_df[['short_name','overall', 'year']], x='year', y='overall', hue='short_name')

ax.set_yticks(np.arange(82,100+1,2))

plt.ylim(80,100)

ax.legend(title='Players');
attrib_categ=['attack_oa', 'skill_oa', 'movements_oa', 'power_oa', 'mentality_oa', 'defending_oa']

categ=['ATT', 'SKI', 'MVT', 'PWR', 'MEN', 'DEF']



def plot_radar(df, p1, p2, att_cat, cat, ax):

    N = len(att_cat)

    range_list = list(np.arange(0,100,20))

    

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    ax.set_theta_offset(7* pi / 6)

    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1],cat)



    ax.set_rlabel_position(0)

    plt.yticks(range_list , map(str, range_list), color="grey", size=10)

    plt.ylim(0,100)



    values_p1 = df.query('short_name == @p1')[attrib_categ].values.flatten().tolist()

    values_p1 += values_p1[:1]

    values_p2 = df.query('short_name == @p2')[attrib_categ].values.flatten().tolist()

    values_p2 += values_p2[:1]



    ax.plot(angles, values_p1, linewidth=1, linestyle='solid', label="group A")

    ax.fill(angles, values_p1, 'b', alpha=0.1)



    ax.plot(angles, values_p2, linewidth=1, linestyle='solid', label="group A")

    ax.fill(angles, values_p2, 'b', alpha=0.1)

plt.figure(figsize=(18,12))

plt.suptitle('C. Ronaldo VS Messi: Attributes Comparaison by year', fontsize='20')

subplot_l = [231,232,233,234,235,236]

for i,j,k in zip(l_df,subplot_l,years):

    ax = plt.subplot(j, polar=True)

    plot_radar(i,'L. Messi', 'Cristiano Ronaldo', attrib_categ, categ, ax)

    plt.title(k)

plt.legend(['L. Messi', 'Cristiano Ronaldo'], title='Players' , bbox_to_anchor=(1.5, 1.2));
df.columns
# Construction of the features dataframe for the regression model



l=[2015,2016,2017,2018,2019]

features = ['age', 'value_eur','wage_eur','international_reputation','skill_moves', 'pace', 'shooting', 'passing', 'dribbling',

            'defending', 'physic', 'attack_oa', 'skill_oa', 'movements_oa', 'power_oa', 'mentality_oa', 

            'defending_oa', 'gk_oa', 'trait_coef','year']



ols_data = df.query('year in @l')[features].copy()

y = df.query('year in @l')['overall'].copy()



ols_data.loc[ols_data.value_eur==0,'value_eur'] = ols_data['value_eur'].mean()

ols_data.loc[ols_data.wage_eur==0,'wage_eur'] = ols_data['wage_eur'].mean()

ols_data['isGK'] = (ols_data['pace']==0).astype(int)

cols = list(ols_data.columns)

ols_data
# Adding 2nd degree polynomial features to make the regression model more accurate



poly = PolynomialFeatures(2, include_bias=False)

ols_data = poly.fit_transform(ols_data)
# Train/Test data split



x_train, x_test, y_train, y_test = train_test_split( ols_data, y, test_size=0.2, shuffle=True, random_state=42)

display((x_train.shape, y_train.shape))

x_test.shape, y_test.shape
# Training the regression model.



regr = linear_model.LinearRegression()

regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)



print('Mean squared error: %.2f' % mean_squared_error(y_test,y_pred))

print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))
# Getting a random player



def get_example(df,features):

    temp = df.sample(1)

    x = temp[features].copy()

    x['isGK'] = (x['pace']==0).astype(int)

    x= poly.fit_transform(x)

    y = np.array(temp['overall'].copy())

    return x,y
# Predicting a random player overall rating



x,y = get_example(df, features)

est_y = regr.predict(x)

print('y=',int(y), 'pred=', int(np.round(est_y)))
x = df20.loc[:,features].copy()

x['isGK'] = (x['pace']==0).astype(int)

x = poly.fit_transform(x)

true_20 = np.array(df20.loc[:,['overall']].copy()).flatten()

pred_20 = np.round(regr.predict(x)).astype(int)
print('Mean squared error: %.2f' % mean_squared_error(true_20, pred_20))

print('Coefficient of determination: %.2f'% r2_score(true_20, pred_20))
plt.figure(figsize=(10,10))

sb.scatterplot(pred_20, true_20, alpha=0.15)

plt.plot(true_20, true_20, color='r', alpha=0.5)

plt.legend(['Correct pred','True-Vs-Pred scatter'])

plt.title('Prediction of overall ratings of FIFA 2020 using OLS model')

plt.xlabel(' Overall Prediction')

plt.ylabel('True Overall');
# Creating categories based on players positions.



reduce_pos = {}

pos = list(df.team_position.unique())

new_pos = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']

map_l = []

for i in pos:

    if i in ['CF','ST','LW','RW']:

        map_l.append((i,'Forward'))

    elif i in ['CM','RM','LM','CDM','CAM']:

        map_l.append((i,'Midfielder'))

    elif i in ['CB','RB','LB','LWB','RWB']:

        map_l.append((i,'Defender'))

    else:

        map_l.append((i,'Goalkeeper'))

for j in map_l:

    reduce_pos[j[0]]=j[1]

reduce_pos
# Selecting features for the classification task.



l=[2015,2016,2017,2018,2019]

features_classification = ['skill_moves', 'pace', 'shooting', 'passing', 'dribbling','defending', 'physic', 

                           'attack_oa', 'skill_oa','movements_oa', 'power_oa', 'mentality_oa', 'defending_oa', 'gk_oa']

classification_data = df.query('year in @l')[features_classification].copy()

y_class_label = df.query('year in @l')['team_position'].copy().map(reduce_pos)

y_class = y_class_label

display(classification_data)

y_class
# Train/Test splitting of the selected data.



X_train, X_test, y_train, y_test = train_test_split(classification_data, y_class, test_size=0.2, random_state=1)

X_train.shape, X_test.shape
# Using cross-validation to tune the max leaf parameter of the decision tree.



param_grid = {'max_leaf_nodes': np.arange(5, 30)}

tree_classifier = DecisionTreeClassifier()

tree_CV = GridSearchCV(tree_classifier, param_grid, cv=5)

tree_CV.fit(X_train, y_train);
#View the accuracy score & the best parameters for the model found using grid search



print('Best score for training data:', tree_CV.best_score_,"\n")

print('Best nbr of neighbors:',tree_CV.best_params_,"\n")
# Select the optimal model and testing it.



tree_classifier = tree_CV.best_estimator_

tree_y_pred = tree_classifier.predict(X_test)
# Check the results of the test.



print("Training set score for decison tree: %f" % tree_classifier.score(X_train , y_train))

print("Testing set score for decision tree: %f" % tree_classifier.score(X_test , y_test))

plot_confusion_matrix(tree_classifier, X_test, y_test, cmap='Blues');
# Plot the decision tree.



plt.figure(figsize=(15,15))

plot_tree(tree_classifier, feature_names=list(classification_data.columns) ,class_names=new_pos, filled=True);
# Check random predictions made on the test set.



l_true=list(y_test)

l_pred=list(tree_y_pred)

temp={

    'true_position': l_true,

    'predicted_position': l_pred

}

result = pd.DataFrame(temp)

result.sample(10)
# Test the decision trained tree model on the FIFA 20 data.



X_20 = df20[features_classification].copy()

y_20 = df20['team_position'].copy().map(reduce_pos)

tree_y20_pred = tree_classifier.predict(X_20)
# Check results



print('Accuracy:', accuracy_score(y_20, tree_y20_pred))

plot_confusion_matrix(tree_classifier, X_20, y_20, labels=['Goalkeeper','Defender','Midfielder','Forward'], cmap='Blues');
# Map positions to integers.



pos_dic={'Goalkeeper': 0, 'Defender': 1, 'Midfielder':2, 'Forward':3}

y_train = y_train.map(pos_dic)

y_test = y_test.map(pos_dic)

y_test
# Use cross-validation to tune the neighbors number parameter of the KNN model. 



param_grid = {'n_neighbors': np.arange(1, 25)}

KNN_classifier = KNeighborsClassifier()

KNN_CV = GridSearchCV(KNN_classifier, param_grid, cv=5)

KNN_CV.fit(X_train, y_train);
#View the accuracy score & best parameters for the model found using grid search.



print('Best score for training data:', KNN_CV.best_score_,"\n")

print('Best nbr of neighbors:',KNN_CV.best_params_,"\n")
# Select the optimal KNN model.



KNN_classifier = KNN_CV.best_estimator_

y_pred = KNN_classifier.predict(X_test)
# View the score on the test set.



print("Training set score for KNN: %f" % KNN_classifier.score(X_train , y_train))

print("Testing set score for KNN: %f" % KNN_classifier.score(X_test , y_test))

plot_confusion_matrix(KNN_classifier, X_test, y_test, display_labels=['Goalkeeper','Defender','Midfielder','Forward'], cmap='Blues');
# Testing the trained KNN model on the FIFA 20 data.



KNN_y20_pred = KNN_classifier.predict(X_20)

y_20_int_label = y_20.map(pos_dic)

print('Accuracy:', accuracy_score(y_20_int_label, KNN_y20_pred))

plot_confusion_matrix(KNN_classifier, X_20, y_20_int_label, display_labels=['Goalkeeper','Defender','Midfielder','Forward'], cmap='Blues');
# Scaling data.



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Optimal parameters.



params_grid = [{'kernel': ['rbf'], 'gamma': [1e-1],'C': [100]}]
# Train and SVM model using the optimal parameters.



optimal_SVM_model = svm.SVC(C=100, kernel='rbf', gamma=1e-1,)

optimal_SVM_model.fit(X_train_scaled, y_train);
# View results.



print("Training set score for SVM: %f" % optimal_SVM_model.score(X_train_scaled , y_train))

print("Testing set score for SVM: %f" % optimal_SVM_model.score(X_test_scaled  , y_test))
# Test the trained model, and view results.



SVM_y_pred = optimal_SVM_model.predict(X_test_scaled)

print('Accuracy:',accuracy_score(y_test, SVM_y_pred))

plot_confusion_matrix(optimal_SVM_model, X_test_scaled, y_test, display_labels=['Goalkeeper','Defender','Midfielder','Forward'], cmap='Blues');
# Generate the classification report.



inv_pos_dic={ 0:'Goalkeeper',  1:'Defender', 2:'Midfielder', 3:'Forward'}

y_test_label = pd.Series(y_test).map(inv_pos_dic)

y_pred_label = pd.Series(SVM_y_pred).map(inv_pos_dic)



print(classification_report(y_test_label,y_pred_label))
# Test the final model on the FIFA 20 data.



X_20_scaled = scaler.fit_transform(X_20)

SVM_y20_pred = optimal_SVM_model.predict(X_20_scaled)

print('Accuracy:', accuracy_score(y_20_int_label, SVM_y20_pred))

plot_confusion_matrix(optimal_SVM_model, X_20_scaled, y_20_int_label, display_labels=['Goalkeeper','Defender','Midfielder','Forward'], cmap='Blues');
acc_tree = accuracy_score(y_20, tree_y20_pred)

acc_KNN = accuracy_score(y_20_int_label, KNN_y20_pred)

acc_SVM = accuracy_score(y_20_int_label, SVM_y20_pred)



r2_tree = r2_score(pd.Series(y_20).map(pos_dic), pd.Series(tree_y20_pred).map(pos_dic))

r2_KNN = r2_score(y_20_int_label, KNN_y20_pred)

r2_SVM = r2_score(y_20_int_label, SVM_y20_pred)



acc_l=[acc_tree, acc_KNN, acc_SVM]

r2_l=[r2_tree, r2_KNN, r2_SVM]

index_l=['Decision_Tree', 'KNN', 'SVM']

df_acc = pd.DataFrame(list(zip(acc_l, r2_l)), index =index_l, columns =['Accuracy', 'R2_score']) 

df_acc
tree_cm = confusion_matrix(y_20, tree_y20_pred, normalize='true')

KNN_cm = confusion_matrix(y_20_int_label, KNN_y20_pred, normalize='true')

SVM_cm = confusion_matrix(y_20_int_label, SVM_y20_pred, normalize='true')



def plot_cm(mat,y_ture,ax,case):

    if case == 0:

        df_cm = pd.DataFrame(mat, columns=np.unique(y_ture), index = np.unique(y_ture))

        df_cm.index.name = 'True Label'

        df_cm.columns.name = 'Predicted Label'

        sb.heatmap(df_cm, cmap="Blues", cbar=False, annot=True,annot_kws={"size": 10}, ax=ax)

        plt.yticks(fontsize=10)

        plt.xticks(fontsize=10)

    else:

        l_lab=['Goalkeeper','Defender','Midfielder','Forward']

        df_cm = pd.DataFrame(mat, columns=np.array(l_lab), index = np.unique(l_lab))

        df_cm.index.name = 'True Label'

        df_cm.columns.name = 'Predicted Label'

        sb.heatmap(df_cm, cmap="Blues", cbar=False, annot=True,annot_kws={"size": 10}, ax=ax)

        plt.yticks(fontsize=10)

        plt.xticks(fontsize=10)
plt.figure(figsize=(20,5))

plt.tight_layout()

ax1 = plt.subplot(131)

plt.title('CM of the Decision Tree Model for FIFA 20 data')

plot_cm(tree_cm, y_20, ax1,0)

ax2 = plt.subplot(132)

plt.title('CM of the KNN Model for FIFA 20 data')

plot_cm(KNN_cm, y_20, ax2,1)

ax3 = plt.subplot(133)

plt.title('CM of the SVM Model for FIFA 20 data')

plot_cm(SVM_cm, y_20, ax3,1)
cols = {'0':'Goalkeeper', '1':'Defender', '2': 'Midfielder', '3':'Forward'}

print('Classification report of the decision tree model for the FIFA 2020 data:')

display(pd.DataFrame(classification_report(y_20, tree_y20_pred, output_dict=True)))

print('Classification report of the KNN model for the FIFA 2020 data:')

display(pd.DataFrame(classification_report(y_20_int_label, KNN_y20_pred, output_dict=True)).rename(columns=cols))

print('Classification report of the SVM model for the FIFA 2020 data:')

display(pd.DataFrame(classification_report(y_20_int_label, SVM_y20_pred, output_dict=True)).rename(columns=cols))