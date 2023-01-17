import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from functools import reduce

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', None)

pd.options.mode.chained_assignment = None



df20 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

df19 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_19.csv')

df18 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_18.csv')

df17 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_17.csv')

df16 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_16.csv')

df15 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_15.csv')
#from the kernel https://www.kaggle.com/laowingkin/fifa-18-find-the-best-squad

#thanks DLao



df20_pot = df20[(df20.age.astype(int) >= 18) & (df20.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df20_over = df20[(df20.age.astype(int) >= 18) & (df20.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df20_summary = pd.concat([df20_pot, df20_over], axis=1)

df19_pot = df19[(df19.age.astype(int) >= 18) & (df19.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df19_over = df19[(df19.age.astype(int) >= 18) & (df19.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df19_summary = pd.concat([df19_pot, df19_over], axis=1)

df18_pot = df18[(df18.age.astype(int) >= 18) & (df18.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df18_over = df18[(df18.age.astype(int) >= 18) & (df18.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df18_summary = pd.concat([df18_pot, df18_over], axis=1)

df17_pot = df17[(df17.age.astype(int) >= 18) & (df17.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df17_over = df17[(df17.age.astype(int) >= 18) & (df17.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df17_summary = pd.concat([df17_pot, df17_over], axis=1)

df16_pot = df16[(df16.age.astype(int) >= 18) & (df16.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df16_over = df16[(df16.age.astype(int) >= 18) & (df16.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df16_summary = pd.concat([df16_pot, df16_over], axis=1)

df15_pot = df15[(df15.age.astype(int) >= 18) & (df15.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df15_over = df15[(df15.age.astype(int) >= 18) & (df15.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df15_summary = pd.concat([df15_pot, df15_over], axis=1)



fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1, figsize=(10, 25))

ax1.plot(df20_summary)

ax1.set_ylabel('Rating')

ax1.set_title('FIFA 20 - Average Rating by Age')

ax2.plot(df19_summary)

ax2.set_ylabel('Rating')

ax2.set_title('FIFA 19 - Average Rating by Age')

ax3.plot(df18_summary)

ax3.set_ylabel('Rating')

ax3.set_title('FIFA 18 - Average Rating by Age')

ax4.plot(df17_summary)

ax4.set_ylabel('Rating')

ax4.set_title('FIFA 17 - Average Rating by Age')

ax5.plot(df16_summary)

ax5.set_ylabel('Rating')

ax5.set_title('FIFA 16 - Average Rating by Age')

ax6.plot(df15_summary)

ax6.set_ylabel('Rating')

ax6.set_title('FIFA 15 - Average Rating by Age')
df20['best_position'] = df20['player_positions'].str.split(',').str[0]

df19['best_position'] = df19['player_positions'].str.split(',').str[0]

df18['best_position'] = df18['player_positions'].str.split(',').str[0]

df17['best_position'] = df17['player_positions'].str.split(',').str[0]

df16['best_position'] = df16['player_positions'].str.split(',').str[0]

df15['best_position'] = df15['player_positions'].str.split(',').str[0]



def get_best_squad(df_name, position):

    df_copy = df_name.copy()

    store = []

    for i in position:

        store.append([i,df_copy.loc[[df_copy[df_copy['best_position'] == i]['overall'].idxmax()]]['short_name'].to_string(index = False), df_copy[df_copy['best_position'] == i]['overall'].max()])

        df_copy.drop(df_copy[df_copy['best_position'] == i]['overall'].idxmax(), inplace = True)

    #return store

    return pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', 'Overall']).to_string(index = False)



# 4-3-3

squad_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

print ('4-3-3 in FIFA 20')

print (get_best_squad(df20, squad_433))

print ('\n4-3-3 in FIFA 19')

print (get_best_squad(df19, squad_433))

print ('\n4-3-3 in FIFA 18')

print (get_best_squad(df18, squad_433))

print ('\n4-3-3 in FIFA 17')

print (get_best_squad(df17, squad_433))

print ('\n4-3-3 in FIFA 16')

print (get_best_squad(df16, squad_433))

print ('\n4-3-3 in FIFA 15')

print (get_best_squad(df15, squad_433))
# 3-5-2

squad_352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']

print ('3-5-2 in FIFA 20')

print (get_best_squad(df20, squad_352))

print ('\n3-5-2 in FIFA 19')

print (get_best_squad(df19, squad_352))

print ('\n3-5-2 in FIFA 18')

print (get_best_squad(df18, squad_352))

print ('\n3-5-2 in FIFA 17')

print (get_best_squad(df17, squad_352))

print ('\n3-5-2 in FIFA 16')

print (get_best_squad(df16, squad_352))

print ('\n3-5-2 in FIFA 15')

print (get_best_squad(df15, squad_352))
df20['value_million_eur'] = pd.to_numeric(df20['value_eur'], errors='coerce') / 1000000

df19['value_million_eur'] = pd.to_numeric(df19['value_eur'], errors='coerce') / 1000000

df18['value_million_eur'] = pd.to_numeric(df18['value_eur'], errors='coerce') / 1000000

df17['value_million_eur'] = pd.to_numeric(df17['value_eur'], errors='coerce') / 1000000

df16['value_million_eur'] = pd.to_numeric(df16['value_eur'], errors='coerce') / 1000000

df15['value_million_eur'] = pd.to_numeric(df15['value_eur'], errors='coerce') / 1000000



def get_best_squad(df_name, position, club = '*', measurement = 'overall'):

    df_copy = df_name.copy()

    df_copy = df_copy[df_copy['club'] == club]

    store = []

    for i in position:

        store.append([df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['best_position'].to_string(index = False),df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['short_name'].to_string(index = False), df_copy[df_copy['best_position'].str.contains(i)][measurement].max(), float(df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['value_million_eur'].to_string(index = False))])

        df_copy.drop(df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax(), inplace = True)

    return np.mean([x[2] for x in store]).round(1), pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', measurement, 'Value (M)']).to_string(index = False), np.sum([x[3] for x in store]).round(1)



# easier constraint

squad_433_adj = ['GK', 'B$', 'B$', 'B$', 'B$', 'M$', 'M$', 'M$', 'W$|T$', 'W$|T$', 'W$|T$']



# Example Output for ManCity

rating_433_ManCity_Overall, best_list_433_ManCity_Overall, value_433_ManCity_Overall = get_best_squad(df20, squad_433_adj, 'Manchester City', 'overall')

rating_433_ManCity_Potential, best_list_433_ManCity_Potential, value_433_ManCity_Potential  = get_best_squad(df20, squad_433_adj, 'Manchester City', 'potential')

print('FIFA 20 - Manchester City\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_433_ManCity_Overall))

print('Total Value (M): {:.1f}'.format(value_433_ManCity_Overall))

print(best_list_433_ManCity_Overall)

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_433_ManCity_Potential))

print('Total Value (M): {:.1f}'.format(value_433_ManCity_Potential))

print(best_list_433_ManCity_Potential)
# Example Output for ManUtd

rating_433_ManUtd_Overall, best_list_433_ManUtd_Overall, value_433_ManUtd_Overall = get_best_squad(df20, squad_433_adj, 'Manchester United', 'overall')

rating_433_ManUtd_Potential, best_list_433_ManUtd_Potential, value_433_ManUtd_Potential  = get_best_squad(df20, squad_433_adj, 'Manchester United', 'potential')

print('FIFA 20 - Manchester United\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_433_ManUtd_Overall))

print('Total Value (M): {:.1f}'.format(value_433_ManUtd_Overall))

print(best_list_433_ManUtd_Overall)

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_433_ManUtd_Potential))

print('Total Value (M): {:.1f}'.format(value_433_ManUtd_Potential))

print(best_list_433_ManUtd_Potential)
def get_top20_players(df_name):

    df_copy = df_name.sort_values(['overall', 'potential'], ascending=[False, False]).head(20)

    store = []

    for index, row in df_copy.iterrows():

        store.append([row['best_position'], row['short_name'], row['overall'], row['potential'], row['age']])

    return np.mean([x[2] for x in store]).round(1), np.mean([x[3] for x in store]).round(1), np.mean([x[4] for x in store]).round(1), pd.DataFrame(np.array(store).reshape(20, 5), columns = ['Position', 'Player', 'Overall', 'Potential', 'Age']).to_string(index = False)



top20_players20_overall, top20_players20_potential, top20_players20_age, top20_players20 = get_top20_players(df20)

print('FIFA 20 - Top 20 Players')

print('Average overall: {:.1f}'.format(top20_players20_overall))

print('Average potential: {:.1f}'.format(top20_players20_potential))

print('Average age: {:.1f}'.format(top20_players20_age))

print(top20_players20)



top20_players19_overall, top20_players19_potential, top20_players19_age, top20_players19 = get_top20_players(df19)

print('\nFIFA 19 - Top 20 Players')

print('Average overall: {:.1f}'.format(top20_players19_overall))

print('Average potential: {:.1f}'.format(top20_players19_potential))

print('Average age: {:.1f}'.format(top20_players19_age))

print(top20_players19)



top20_players18_overall, top20_players18_potential, top20_players18_age, top20_players18 = get_top20_players(df18)

print('\nFIFA 18 - Top 20 Players')

print('Average overall: {:.1f}'.format(top20_players18_overall))

print('Average potential: {:.1f}'.format(top20_players18_potential))

print('Average age: {:.1f}'.format(top20_players18_age))

print(top20_players18)



top20_players17_overall, top20_players17_potential, top20_players17_age, top20_players17 = get_top20_players(df17)

print('\nFIFA 17 - Top 20 Players')

print('Average overall: {:.1f}'.format(top20_players17_overall))

print('Average potential: {:.1f}'.format(top20_players17_potential))

print('Average age: {:.1f}'.format(top20_players17_age))

print(top20_players17)



top20_players16_overall, top20_players16_potential, top20_players16_age, top20_players16 = get_top20_players(df16)

print('\nFIFA 16 - Top 20 Players')

print('Average overall: {:.1f}'.format(top20_players16_overall))

print('Average potential: {:.1f}'.format(top20_players16_potential))

print('Average age: {:.1f}'.format(top20_players16_age))

print(top20_players16)



top20_players15_overall, top20_players15_potential, top20_players15_age, top20_players15 = get_top20_players(df15)

print('\nFIFA 15 - Top 20 Players')

print('Average overall: {:.1f}'.format(top20_players15_overall))

print('Average potential: {:.1f}'.format(top20_players15_potential))

print('Average age: {:.1f}'.format(top20_players15_age))

print(top20_players15)
#['Lionel Andrés Messi Cuccittini', 'Cristiano Ronaldo dos Santos Aveiro', 'Neymar da Silva Santos Junior', 'Jan Oblak', 'Eden Hazard', 'Kevin De Bruyne', 'Marc-André ter Stegen', 'Virgil van Dijk', 'Luka Modrić', 'Mohamed  Salah Ghaly', 'Kylian Mbappé', 'Kalidou Koulibaly', 'Harry Kane', 'Alisson Ramses Becker', 'David De Gea Quintana', "N'Golo Kanté", 'Giorgio Chiellini', 'Sergio Leonel Agüero del Castillo', 'Sergio Ramos García', 'Luis Alberto Suárez Díaz']

top20_players20_list = df20.sort_values(['overall', 'potential'], ascending=[False, False]).head(20)['long_name'].tolist()

filtered_players20 = df20[df20['long_name'].isin(top20_players20_list)][['short_name', 'overall', 'potential']]

filtered_players20.columns = ['short_name', 'overall20', 'potential20']

filtered_players20.set_index('short_name', inplace=True)

filtered_players19 = df19[df19['long_name'].isin(top20_players20_list)][['short_name', 'overall', 'potential']]

filtered_players19.columns = ['short_name', 'overall19', 'potential19']

filtered_players19.set_index('short_name', inplace=True)

filtered_players18 = df18[df18['long_name'].isin(top20_players20_list)][['short_name', 'overall', 'potential']]

filtered_players18.columns = ['short_name', 'overall18', 'potential18']

filtered_players18.set_index('short_name', inplace=True)

top20_players20_list = [item.replace('Kylian Mbappé', 'Kylian Mbappe Lottin') for item in top20_players20_list] #Mbappe had a different name in FIFA 17

filtered_players17 = df17[df17['long_name'].isin(top20_players20_list)][['short_name', 'overall', 'potential']]

filtered_players17.at[3899, 'short_name'] = 'Kylian Mbappé'

filtered_players17.columns = ['short_name', 'overall17', 'potential17']

filtered_players17.set_index('short_name', inplace=True)

filtered_players16 = df16[df16['long_name'].isin(top20_players20_list)][['short_name', 'overall', 'potential']] #VVD was not included in the sofifa database as of 21.09.2015 (not anymore in Celtic, not yet in Southamption)

filtered_players16.columns = ['short_name', 'overall16', 'potential16']

filtered_players16.set_index('short_name', inplace=True)

filtered_players15 = df15[df15['long_name'].isin(top20_players20_list)][['short_name', 'overall', 'potential']]

filtered_players15.columns = ['short_name', 'overall15', 'potential15']

filtered_players15.set_index('short_name', inplace=True)



dfs = [filtered_players20, filtered_players19, filtered_players18, filtered_players17, filtered_players16, filtered_players15]

merged_df = reduce(lambda left,right: pd.merge(left, right, on='short_name'), dfs)

overall_df = merged_df[['overall20', 'overall19', 'overall18', 'overall17', 'overall16', 'overall15']]

overall_df.columns = ['FIFA 20', 'FIFA 19', 'FIFA 18', 'FIFA 17', 'FIFA 16', 'FIFA 15']

potential_df = merged_df[['potential20', 'potential19', 'potential18', 'potential17', 'potential16', 'potential15']]

potential_df.columns = ['FIFA 20', 'FIFA 19', 'FIFA 18', 'FIFA 17', 'FIFA 16', 'FIFA 15']
transposed_overall_df = overall_df.T.sort_index(ascending=True)

plt.rcParams["figure.figsize"] = [15, 10]

ax = transposed_overall_df.plot()

ax.set_ylabel('Overall Rating')

ax.set_title('FIFA 20 - Top 20 Players Historical Overall Rating')
transposed_potential_df = potential_df.T.sort_index(ascending=True)

plt.rcParams["figure.figsize"] = [15, 10]

ax = transposed_potential_df.plot()

ax.set_ylabel('Potential Rating')

ax.set_title('FIFA 20 - Top 20 Players Historical Potential Rating')