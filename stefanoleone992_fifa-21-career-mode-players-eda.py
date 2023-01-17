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



df21 = pd.read_csv('/kaggle/input/fifa-21-complete-player-dataset/players_21.csv')

df20 = pd.read_csv('/kaggle/input/fifa-21-complete-player-dataset/players_20.csv')

df19 = pd.read_csv('/kaggle/input/fifa-21-complete-player-dataset/players_19.csv')

df18 = pd.read_csv('/kaggle/input/fifa-21-complete-player-dataset/players_18.csv')

df17 = pd.read_csv('/kaggle/input/fifa-21-complete-player-dataset/players_17.csv')

df16 = pd.read_csv('/kaggle/input/fifa-21-complete-player-dataset/players_16.csv')

df15 = pd.read_csv('/kaggle/input/fifa-21-complete-player-dataset/players_15.csv')
#from the kernel https://www.kaggle.com/laowingkin/fifa-18-find-the-best-squad

#thanks DLao



df21_pot = df21[(df21.age.astype(int) >= 18) & (df21.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df21_over = df21[(df21.age.astype(int) >= 18) & (df21.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df21_summary = pd.concat([df21_pot, df21_over], axis=1)

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



fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows=7, ncols=1, figsize=(10, 30))

ax1.plot(df21_summary)

ax1.set_ylabel('Rating')

ax1.set_title('FIFA 21 - Average Rating by Age')

ax2.plot(df20_summary)

ax2.set_ylabel('Rating')

ax2.set_title('FIFA 20 - Average Rating by Age')

ax3.plot(df19_summary)

ax3.set_ylabel('Rating')

ax3.set_title('FIFA 19 - Average Rating by Age')

ax4.plot(df18_summary)

ax4.set_ylabel('Rating')

ax4.set_title('FIFA 18 - Average Rating by Age')

ax5.plot(df17_summary)

ax5.set_ylabel('Rating')

ax5.set_title('FIFA 17 - Average Rating by Age')

ax6.plot(df16_summary)

ax6.set_ylabel('Rating')

ax6.set_title('FIFA 16 - Average Rating by Age')

ax7.plot(df15_summary)

ax7.set_ylabel('Rating')

ax7.set_title('FIFA 15 - Average Rating by Age')
# adding the 'best_position' field to each df

df21['best_position'] = df21['player_positions'].str.split(',').str[0]

df20['best_position'] = df20['player_positions'].str.split(',').str[0]

df19['best_position'] = df19['player_positions'].str.split(',').str[0]

df18['best_position'] = df18['player_positions'].str.split(',').str[0]

df17['best_position'] = df17['player_positions'].str.split(',').str[0]

df16['best_position'] = df16['player_positions'].str.split(',').str[0]

df15['best_position'] = df15['player_positions'].str.split(',').str[0]



# adding the 'value_million_eur' field to each df

df21['value_million_eur'] = pd.to_numeric(df21['value_eur'], errors='coerce') / 1000000

df20['value_million_eur'] = pd.to_numeric(df20['value_eur'], errors='coerce') / 1000000

df19['value_million_eur'] = pd.to_numeric(df19['value_eur'], errors='coerce') / 1000000

df18['value_million_eur'] = pd.to_numeric(df18['value_eur'], errors='coerce') / 1000000

df17['value_million_eur'] = pd.to_numeric(df17['value_eur'], errors='coerce') / 1000000

df16['value_million_eur'] = pd.to_numeric(df16['value_eur'], errors='coerce') / 1000000

df15['value_million_eur'] = pd.to_numeric(df15['value_eur'], errors='coerce') / 1000000



def get_dream_team(df_name, position, measurement = 'overall'):

    df_copy = df_name.copy()

    store = []

    for i in position:

        store.append([df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['best_position'].to_string(index = False),

                      df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['short_name'].to_string(index = False),

                      df_copy[df_copy['best_position'].str.contains(i)][measurement].max(),

                      float(df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['value_million_eur'].to_string(index = False)),

                      df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['club_name'].to_string(index = False)])

        df_copy.drop(df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax(), inplace = True)

    print('Average rating: {:.1f}'.format(np.mean([x[2] for x in store]).round(1)))

    print('Total Value (M): {:.1f}'.format(np.sum([x[3] for x in store]).round(1)))

    return pd.DataFrame(np.array(store).reshape(11, 5), columns = ['Position', 'Player', measurement, 'Value (M)', 'Club']).to_string(index = False)

    

# 4-3-3

squad_433 = ['GK', 'LB|LWB|LDM', 'LCB|CB', 'RCB|CB', 'RB|RWB|RDM', 'CDM|CM|CAM', 'CDM|CM|CAM', 'CDM|CM|CAM', 'LF|LW|CF|ST', 'CF|ST', 'RF|RW|CF|ST']



print ('4-3-3 Dream Team in FIFA 21')

print (get_dream_team(df21, squad_433))

print ('\n4-3-3 Dream Team in FIFA 20')

print (get_dream_team(df20, squad_433))

print ('\n4-3-3 Dream Team in FIFA 19')

print (get_dream_team(df19, squad_433))

print ('\n4-3-3 Dream Team in FIFA 18')

print (get_dream_team(df18, squad_433))

print ('\n4-3-3 Dream Team in FIFA 17')

print (get_dream_team(df17, squad_433))

print ('\n4-3-3 Dream Team in FIFA 16')

print (get_dream_team(df16, squad_433))

print ('\n4-3-3 Dream Team in FIFA 15')

print (get_dream_team(df15, squad_433))
# 3-5-2

squad_352 = ['GK', 'LB|LCB|CB', 'CB', 'RB|RCB|CB', 'LW|LWB|LM|LCM', 'CDM|CM', 'CDM|CM', 'CM|CAM|CF', 'RW|RWB|RM|RCM', 'LC|LW|ST', 'RC|RW|ST']



print ('3-5-2 Dream Team in FIFA 21')

print (get_dream_team(df21, squad_352))

print ('\n3-5-2 Dream Team in FIFA 20')

print (get_dream_team(df20, squad_352))

print ('\n3-5-2 Dream Team in FIFA 19')

print (get_dream_team(df19, squad_352))

print ('\n3-5-2 Dream Team in FIFA 18')

print (get_dream_team(df18, squad_352))

print ('\n3-5-2 Dream Team in FIFA 17')

print (get_dream_team(df17, squad_352))

print ('\n3-5-2 Dream Team in FIFA 16')

print (get_dream_team(df16, squad_352))

print ('\n3-5-2 Dream Team in FIFA 15')

print (get_dream_team(df15, squad_352))
def get_best_squad(df_name, club = '*', measurement = 'overall'):

    df_copy = df_name.copy()

    df_copy = df_copy[df_copy['club_name'] == club]

    store_433 = []

    for i in squad_433:

        store_433.append([df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['best_position'].to_string(index = False),

                      df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['short_name'].to_string(index = False), 

                      df_copy[df_copy['best_position'].str.contains(i)][measurement].max(), 

                      float(df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['value_million_eur'].to_string(index = False))])

        df_copy.drop(df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax(), inplace = True)

    store_352 = []

    df_copy = df_name.copy()

    df_copy = df_copy[df_copy['club_name'] == club]

    for i in squad_352:

        store_352.append([df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['best_position'].to_string(index = False),

                      df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['short_name'].to_string(index = False), 

                      df_copy[df_copy['best_position'].str.contains(i)][measurement].max(), 

                      float(df_copy.loc[[df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax()]]['value_million_eur'].to_string(index = False))])

        df_copy.drop(df_copy[df_copy['best_position'].str.contains(i)][measurement].idxmax(), inplace = True)

    if np.mean([x[2] for x in store_433]).round(1) >= np.mean([x[2] for x in store_352]).round(1):

        return np.mean([x[2] for x in store_433]).round(1), pd.DataFrame(np.array(store_433).reshape(11,4), columns = ['Position', 'Player', measurement, 'Value (M)']).to_string(index = False), np.sum([x[3] for x in store_433]).round(1), '4-3-3'

    else:

        return np.mean([x[2] for x in store_352]).round(1), pd.DataFrame(np.array(store_352).reshape(11,4), columns = ['Position', 'Player', measurement, 'Value (M)']).to_string(index = False), np.sum([x[3] for x in store_352]).round(1), '3-5-2'
# RealMadrid



rating_RealMadrid_Overall, best_list_RealMadrid_Overall, value_RealMadrid_Overall, formation_RealMadrid_Overall = get_best_squad(df21, 'Real Madrid', 'overall')

print('FIFA 21 - Real Madrid\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_RealMadrid_Overall))

print('Total Value (M): {:.1f}'.format(value_RealMadrid_Overall))

print('Squad Formation: ' + formation_RealMadrid_Overall)

print(best_list_RealMadrid_Overall)



rating_RealMadrid_Potential, best_list_RealMadrid_Potential, value_RealMadrid_Potential, formation_RealMadrid_Potential = get_best_squad(df21, 'Real Madrid', 'potential')

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_RealMadrid_Potential))

print('Total Value (M): {:.1f}'.format(value_RealMadrid_Potential))

print('Squad Formation: ' + formation_RealMadrid_Potential)

print(best_list_RealMadrid_Potential)



# Barcelona

rating_Barcelona_Overall, best_list_Barcelona_Overall, value_Barcelona_Overall, formation_Barcelona_Overall = get_best_squad(df21, 'FC Barcelona', 'overall')

print('\n\n\nFIFA 21 - FC Barcelona\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_Barcelona_Overall))

print('Total Value (M): {:.1f}'.format(value_Barcelona_Overall))

print('Squad Formation: ' + formation_Barcelona_Overall)

print(best_list_Barcelona_Overall)



rating_Barcelona_Potential, best_list_Barcelona_Potential, value_Barcelona_Potential, formation_Barcelona_Potential = get_best_squad(df21, 'FC Barcelona', 'potential')

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_Barcelona_Potential))

print('Total Value (M): {:.1f}'.format(value_Barcelona_Potential))

print('Squad Formation: ' + formation_Barcelona_Potential)

print(best_list_Barcelona_Potential)
# Liverpool



rating_Liverpool_Overall, best_list_Liverpool_Overall, value_Liverpool_Overall, formation_Liverpool_Overall = get_best_squad(df21, 'Liverpool', 'overall')

print('FIFA 21 - Liverpool\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_Liverpool_Overall))

print('Total Value (M): {:.1f}'.format(value_Liverpool_Overall))

print('Squad Formation: ' + formation_Liverpool_Overall)

print(best_list_Liverpool_Overall)



rating_Liverpool_Potential, best_list_Liverpool_Potential, value_Liverpool_Potential, formation_Liverpool_Potential = get_best_squad(df21, 'Liverpool', 'potential')

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_Liverpool_Potential))

print('Total Value (M): {:.1f}'.format(value_Liverpool_Potential))

print('Squad Formation: ' + formation_Liverpool_Potential)

print(best_list_Liverpool_Potential)



# ManCity

rating_ManCity_Overall, best_list_ManCity_Overall, value_ManCity_Overall, formation_ManCity_Overall = get_best_squad(df21, 'Manchester City', 'overall')

print('\n\n\nFIFA 21 - Manchester City\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_ManCity_Overall))

print('Total Value (M): {:.1f}'.format(value_ManCity_Overall))

print('Squad Formation: ' + formation_ManCity_Overall)

print(best_list_ManCity_Overall)



rating_ManCity_Potential, best_list_ManCity_Potential, value_ManCity_Potential, formation_ManCity_Potential = get_best_squad(df21, 'Manchester City', 'potential')

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_ManCity_Potential))

print('Total Value (M): {:.1f}'.format(value_ManCity_Potential))

print('Squad Formation: ' + formation_ManCity_Potential)

print(best_list_ManCity_Potential)
# Juventus



rating_Juventus_Overall, best_list_Juventus_Overall, value_Juventus_Overall, formation_Juventus_Overall = get_best_squad(df21, 'Juventus', 'overall')

print('FIFA 21 - Juventus\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_Juventus_Overall))

print('Total Value (M): {:.1f}'.format(value_Juventus_Overall))

print('Squad Formation: ' + formation_Juventus_Overall)

print(best_list_Juventus_Overall)



rating_Juventus_Potential, best_list_Juventus_Potential, value_Juventus_Potential, formation_Juventus_Potential = get_best_squad(df21, 'Juventus', 'potential')

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_Juventus_Potential))

print('Total Value (M): {:.1f}'.format(value_Juventus_Potential))

print('Squad Formation: ' + formation_Juventus_Potential)

print(best_list_Juventus_Potential)



# Inter

rating_Inter_Overall, best_list_Inter_Overall, value_Inter_Overall, formation_Inter_Overall = get_best_squad(df21, 'Inter', 'overall')

print('\n\n\nFIFA 21 - Inter\n')

print('-Overall-')

print('Average rating: {:.1f}'.format(rating_Inter_Overall))

print('Total Value (M): {:.1f}'.format(value_Inter_Overall))

print('Squad Formation: ' + formation_Inter_Overall)

print(best_list_Inter_Overall)



rating_Inter_Potential, best_list_Inter_Potential, value_Inter_Potential, formation_Inter_Potential = get_best_squad(df21, 'Inter', 'potential')

print('\n-Potential-')

print('Average rating: {:.1f}'.format(rating_Inter_Potential))

print('Total Value (M): {:.1f}'.format(value_Inter_Potential))

print('Squad Formation: ' + formation_Inter_Potential)

print(best_list_Inter_Potential)
def get_top10_players(df_name):

    df_copy = df_name.sort_values(['overall', 'potential'], ascending=[False, False]).head(10)

    store = []

    for index, row in df_copy.iterrows():

        store.append([row['best_position'], row['short_name'], row['overall'], row['potential'], row['age']])

    return np.mean([x[2] for x in store]).round(1), np.mean([x[3] for x in store]).round(1), np.mean([x[4] for x in store]).round(1), pd.DataFrame(np.array(store).reshape(10, 5), columns = ['Position', 'Player', 'Overall', 'Potential', 'Age']).to_string(index = False)



top10_players21_overall, top10_players21_potential, top10_players21_age, top10_players21 = get_top10_players(df21)

print('FIFA 21 - Top 10 Players')

print('Average overall: {:.1f}'.format(top10_players21_overall))

print('Average potential: {:.1f}'.format(top10_players21_potential))

print('Average age: {:.1f}'.format(top10_players21_age))

print(top10_players21)



top10_players20_overall, top10_players20_potential, top10_players20_age, top10_players20 = get_top10_players(df20)

print('\nFIFA 20 - Top 10 Players')

print('Average overall: {:.1f}'.format(top10_players20_overall))

print('Average potential: {:.1f}'.format(top10_players20_potential))

print('Average age: {:.1f}'.format(top10_players20_age))

print(top10_players20)



top10_players19_overall, top10_players19_potential, top10_players19_age, top10_players19 = get_top10_players(df19)

print('\nFIFA 19 - Top 10 Players')

print('Average overall: {:.1f}'.format(top10_players19_overall))

print('Average potential: {:.1f}'.format(top10_players19_potential))

print('Average age: {:.1f}'.format(top10_players19_age))

print(top10_players19)



top10_players18_overall, top10_players18_potential, top10_players18_age, top10_players18 = get_top10_players(df18)

print('\nFIFA 18 - Top 10 Players')

print('Average overall: {:.1f}'.format(top10_players18_overall))

print('Average potential: {:.1f}'.format(top10_players18_potential))

print('Average age: {:.1f}'.format(top10_players18_age))

print(top10_players18)



top10_players17_overall, top10_players17_potential, top10_players17_age, top10_players17 = get_top10_players(df17)

print('\nFIFA 17 - Top 10 Players')

print('Average overall: {:.1f}'.format(top10_players17_overall))

print('Average potential: {:.1f}'.format(top10_players17_potential))

print('Average age: {:.1f}'.format(top10_players17_age))

print(top10_players17)



top10_players16_overall, top10_players16_potential, top10_players16_age, top10_players16 = get_top10_players(df16)

print('\nFIFA 16 - Top 10 Players')

print('Average overall: {:.1f}'.format(top10_players16_overall))

print('Average potential: {:.1f}'.format(top10_players16_potential))

print('Average age: {:.1f}'.format(top10_players16_age))

print(top10_players16)



top10_players15_overall, top10_players15_potential, top10_players15_age, top10_players15 = get_top10_players(df15)

print('\nFIFA 15 - Top 10 Players')

print('Average overall: {:.1f}'.format(top10_players15_overall))

print('Average potential: {:.1f}'.format(top10_players15_potential))

print('Average age: {:.1f}'.format(top10_players15_age))

print(top10_players15)
top_players21_list = df21.loc[(df21['overall'] >= 90)].sort_values(['overall', 'potential'], ascending=[False, False])['long_name'].tolist()



# Mbappé will not be included in the chart as was not available in FIFA 15 and 16

['Lionel Andrés Messi Cuccittini',

 'Cristiano Ronaldo dos Santos Aveiro',

 'Jan Oblak',

 'Robert Lewandowski',

 'Neymar da Silva Santos Júnior',

 'Kevin De Bruyne',

 'Kylian Mbappé Lottin',

 'Marc-André ter Stegen',

 'Virgil van Dijk',

 'Alisson Ramsés Becker',

 'Sadio Mané',

 'Mohamed Salah Ghaly']



# manually amending the Neymar short_name on FIFA 20-21

df20.loc[df20['short_name'] == 'Neymar Jr', 'short_name'] = 'Neymar'

df21.loc[df21['short_name'] == 'Neymar Jr', 'short_name'] = 'Neymar'



#VVD was not included in the sofifa database as of 2015-09-21 (not anymore in the Celtic's squad, not yet in the Southamption's squad)

if df16.index.max() == 15622:

    df16.loc[df16.index.max() + 1] = 203376

df16.loc[df16['sofifa_id'] == 203376, 'short_name'] = 'V. van Dijk'

df16.loc[df16['sofifa_id'] == 203376, 'long_name'] = 'Virgil van Dijk'

df16.loc[df16['sofifa_id'] == 203376, 'overall'] = 77

df16.loc[df16['sofifa_id'] == 203376, 'potential'] = 83



filtered_players21 = df21[df21['long_name'].isin(top_players21_list)][['short_name', 'overall', 'potential']]

filtered_players21.columns = ['short_name', 'overall21', 'potential21']

filtered_players21.set_index('short_name', inplace=True)

filtered_players20 = df20[df20['long_name'].isin(top_players21_list)][['short_name', 'overall', 'potential']]

filtered_players20.columns = ['short_name', 'overall20', 'potential20']

filtered_players20.set_index('short_name', inplace=True)

filtered_players19 = df19[df19['long_name'].isin(top_players21_list)][['short_name', 'overall', 'potential']]

filtered_players19.columns = ['short_name', 'overall19', 'potential19']

filtered_players19.set_index('short_name', inplace=True)

filtered_players18 = df18[df18['long_name'].isin(top_players21_list)][['short_name', 'overall', 'potential']]

filtered_players18.columns = ['short_name', 'overall18', 'potential18']

filtered_players18.set_index('short_name', inplace=True)

filtered_players17 = df17[df17['long_name'].isin(top_players21_list)][['short_name', 'overall', 'potential']]

filtered_players17.columns = ['short_name', 'overall17', 'potential17']

filtered_players17.set_index('short_name', inplace=True)

filtered_players16 = df16[df16['long_name'].isin(top_players21_list)][['short_name', 'overall', 'potential']]

filtered_players16.columns = ['short_name', 'overall16', 'potential16']

filtered_players16.set_index('short_name', inplace=True)

filtered_players15 = df15[df15['long_name'].isin(top_players21_list)][['short_name', 'overall', 'potential']]

filtered_players15.columns = ['short_name', 'overall15', 'potential15']

filtered_players15.set_index('short_name', inplace=True)



dfs = [filtered_players21, filtered_players20, filtered_players19, filtered_players18, filtered_players17, filtered_players16, filtered_players15]

merged_df = reduce(lambda left,right: pd.merge(left, right, on='short_name'), dfs)

overall_df = merged_df[['overall21', 'overall20', 'overall19', 'overall18', 'overall17', 'overall16', 'overall15']]

overall_df.columns = ['FIFA 21', 'FIFA 20', 'FIFA 19', 'FIFA 18', 'FIFA 17', 'FIFA 16', 'FIFA 15']

potential_df = merged_df[['potential21', 'potential20', 'potential19', 'potential18', 'potential17', 'potential16', 'potential15']]

potential_df.columns = ['FIFA 21', 'FIFA 20', 'FIFA 19', 'FIFA 18', 'FIFA 17', 'FIFA 16', 'FIFA 15']
transposed_overall_df = overall_df.T.sort_index(ascending=True)

plt.rcParams["figure.figsize"] = [20, 12]

ax = transposed_overall_df.plot()

ax.set_ylabel('Overall Rating')

ax.set_title('FIFA 21 - Top Players Historical Overall Rating')
transposed_potential_df = potential_df.T.sort_index(ascending=True)

plt.rcParams["figure.figsize"] = [20, 12]

ax = transposed_potential_df.plot()

ax.set_ylabel('Potential Rating')

ax.set_title('FIFA 21 - Top Players Historical Potential Rating')