import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import matplotlib.cm as cm

import re

sns.set_style("whitegrid")

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('../input/fifa19/data.csv')

df.columns
df = df[['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Position']]

df.head(10)
# get remaining potential

df['Remaining Potential'] = df['Potential'] - df['Overall']
df.head(10)
df['Unit'] = df['Value'].str[-1]

df['Value (M)'] = np.where(df['Unit'] == '0', 0, df['Value'].str[1:-1].replace(r'[a-zA-Z]',''))

df['Value (M)'] = df['Value (M)'].astype(float)

df['Value (M)'] = np.where(df['Unit'] == 'M', df['Value (M)'], df['Value (M)']/1000)
df.head(10)
df = df.drop('Unit', axis = 1)
df.head(10)
# 'ST', 'RW', 'LW', 'GK', 'CDM', 'CB', 'RM', 'CM', 'LM', 'LB', 'CAM','RB', 'CF', 'RWB', 'LWB'



def get_best_squad(position):

    df_copy = df.copy()

    store = []

    for i in position:

        

        store.append([i,df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Position'] == i]['Overall'].max(),df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Club'].to_string(index = False)])

       

        df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace = True)

    #return store

    return pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', 'Overall','Club']).to_string(index = False)



# 4-3-3

squad_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

print ('4-3-3')

print (get_best_squad(squad_433))
# 3-5-2

squad_352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']

print ('3-5-2')

print (get_best_squad(squad_352))
df_p = df.groupby(['Age'])['Potential'].mean()

df_o = df.groupby(['Age'])['Overall'].mean()



df_summary = pd.concat([df_p, df_o], axis=1)



ax = df_summary.plot()

ax.set_ylabel('Rating')

ax.set_title('Average Rating by Age')
df.fillna('',inplace=True)
df.info()
def get_best_squads(position, club = '*', measurement = 'Overall'):

    df_copy = df.copy()

    df_copy = df_copy[df_copy['Club'] == club]

    store = []

    for i in position:

        store.append([df_copy.loc[[df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax()]]['Position'].to_string(index = False),

                      df_copy.loc[[df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax()]]['Name'].to_string(index = False),

                      df_copy[df_copy['Position'].str.contains(i)][measurement].max(), 

                      float(df_copy.loc[[df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax()]]['Value (M)'].to_string(index = False))])

        df_copy.drop(df_copy[df_copy['Position'].str.contains(i)][measurement].idxmax(), inplace = True)

    return np.mean([x[2] for x in store]).round(1),pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', measurement, 'Value (M)']).to_string(index = False),np.sum([x[3] for x in store]).round(1)



# easier constraint

squad_433_adj = ['GK', 'B$', 'B$', 'B$', 'B$', 'M$', 'M$', 'M$', 'W$|T$', 'W$|T$', 'W$|T$']



# Example Output for Chelsea

rating_433_Chelsea_Overall, best_list_433_Chelsea_Overall, value_433_Chelsea_Overall = get_best_squads(squad_433_adj, 'Chelsea', 'Overall')

rating_433_Chelsea_Potential, best_list_433_Chelsea_Potential, value_433_Chelsea_Potential  = get_best_squads(squad_433_adj, 'Chelsea', 'Potential')



print('-Overall-')

print('Average rating: {:.1f}'.format(rating_433_Chelsea_Overall))

print('Total Value (M): {:.1f}'.format(value_433_Chelsea_Overall))

print(best_list_433_Chelsea_Overall)



print('-Potential-')

print('Average rating: {:.1f}'.format(rating_433_Chelsea_Potential))

print('Total Value (M): {:.1f}'.format(value_433_Chelsea_Potential))

print(best_list_433_Chelsea_Potential)

# very easy constraint since some club do not have strict squad

squad_352_adj = ['GK', 'B$', 'B$', 'B$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'W$|T$|M$', 'W$|T$|M$']



By_club = df.groupby(['Club'])['Overall'].mean()



def get_summary(squad):

    OP = []

    # only get top 100 clubs for shorter run-time

    for i in By_club.sort_values(ascending = False).index[0:100]:

        # for overall rating

        O_temp_rating, _, _  = get_best_squads(squad, club = i, measurement = 'Overall')

        # for potential rating & corresponding value

        P_temp_rating, _, P_temp_value = get_best_squads(squad, club = i, measurement = 'Potential')

        OP.append([i, O_temp_rating, P_temp_rating, P_temp_value])

    return OP





OP_df = pd.DataFrame(np.array(get_summary(squad_352_adj)).reshape(-1,4), columns = ['Club', 'Overall', 'Potential', 'Value of highest Potential squad'])

OP_df.set_index('Club', inplace = True)

OP_df = OP_df.astype(float)    
fig, ax = plt.subplots()

OP_df.plot(kind = 'scatter', x = 'Overall', y = 'Potential', c = 'Value of highest Potential squad', 

           s = 50, figsize = (15,15), xlim = (70, 90), ylim = (70, 90),

           title = 'Current Rating vs Potential Rating by Club: 3-5-2', ax = ax)

fig, ax = plt.subplots()

OP_df.plot(kind = 'scatter', x = 'Overall', y = 'Potential', c = 'Value of highest Potential squad',

           s = 50, figsize = (15,15), xlim = (80, 90), ylim = (85, 90),

           title = 'Current Rating vs Potential Rating by Club: 3-5-2', ax = ax)



def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x'], point['y'], str(point['val']))



OP_df['Club_label'] = OP_df.index

        

OP_df_sub = OP_df[(OP_df['Potential']>=85) & (OP_df['Value of highest Potential squad']<=350)]



label_point(OP_df_sub['Overall'], OP_df_sub['Potential'], OP_df_sub['Club_label'], ax)
squad_352_adj = ['GK', 'B$', 'B$', 'B$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'W$|T$|M$', 'W$|T$|M$']



rating_352_TH_Overall, best_list_352_TH_Overall, value_352_TH_Overall = get_best_squads(squad_352_adj, 

                                                                                        'Tottenham Hotspur',

                                                                                        'Overall')

rating_352_TH_Potential, best_list_352_TH_Potential, value_352_TH_Potential  = get_best_squads(squad_352_adj,

                                                                                               'Tottenham Hotspur',

                                                                                               'Potential')

print('Tottenham Hotspur')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_352_TH_Overall))

print('Total Value (M): {:.1f}'.format(value_352_TH_Overall))

print(best_list_352_TH_Overall)



print('-Potential-')

print('Average rating: {:.1f}'.format(rating_352_TH_Potential))

print('Total Value (M): {:.1f}'.format(value_352_TH_Potential))

print(best_list_352_TH_Potential)

# group the data by football club

data_group_by_club = df.groupby('Club')

# find the mean of each attribute and select the Overall column

clubs_average_overall = data_group_by_club.mean()['Overall']

# sort the average overall in descending order and slice the top 5

top_clubs_top_5 = clubs_average_overall.sort_values(ascending = False)[:5]

# filter the big dataframe to include only players from top clubs

fifa18_top_5 = df.loc[df['Club'].isin(top_clubs_top_5.index)]

# create seaborn FacetGrid object, it will contain cell per club

g = sns.FacetGrid(fifa18_top_5, col='Club')

# In each column plot the age distrubtion of a club

g.map(sns.distplot, "Age")

plt.show()
g = sns.FacetGrid(fifa18_top_5, col='Club')

g.map(sns.boxplot, "Age", order='')

plt.show()
Ori_df = pd.read_csv('../input/fifa19/data.csv')

Ori_df.columns
columns_needed_rearranged = ['Aggression','Crossing', 'Curve', 'Dribbling', 'Finishing',

       'FKAccuracy', 'HeadingAccuracy', 'LongShots','Penalties', 'ShotPower', 'Volleys', 

       'ShortPassing', 'LongPassing',

       'Interceptions', 'Marking', 'SlidingTackle', 'StandingTackle',

       'Strength', 'Vision', 'Acceleration', 'Agility', 

       'Reactions', 'Stamina', 'Balance', 'BallControl','Composure','Jumping', 

       'SprintSpeed', 'Positioning','Position']
df_new = Ori_df[columns_needed_rearranged]

df_new.fillna('',inplace = True)

df_new.info()

df_new['Position'] = df_new['Position'].str.strip()

df_new = df_new[df_new['Position'] != 'GK']

df_new.head()

df_new.isnull().values.any()
df_new
fig, ax = plt.subplots()

df_new_ST = df_new[df_new['Position'] == 'ST'].iloc[::200,:-1]

df_new_ST.T.plot.line(color = 'black', figsize = (15,10), legend = False, ylim = (0, 110), title = "ST's attributes distribution", ax=ax)



ax.set_xlabel('Attributes')

ax.set_ylabel('Rating')



for ln in ax.lines:

    ln.set_linewidth(1)



ax.axvline(0, color='red', linestyle='--')   

ax.axvline(12.9, color='red', linestyle='--')



ax.axvline(13, color='blue', linestyle='--')

ax.axvline(17, color='blue', linestyle='--')



ax.axvline(17.1, color='green', linestyle='--')

ax.axvline(28, color='green', linestyle='--')



ax.text(5, 100, 'Attack Attributes', color = 'red', weight = 'bold')

ax.text(13.5, 100, 'Defend Attributes', color = 'blue', weight = 'bold')

ax.text(22, 100, 'Mixed Attributes', color = 'green', weight = 'bold')