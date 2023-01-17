#import modules



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn import preprocessing
#Read in and take a look at data



All_stats = pd.read_csv('../input/nfl-all-stats/Career_Stats_All_Stats.csv')



print (All_stats.head(10))

print (All_stats.columns)
#clean data and only consider data starting in Super Bowl Era



All_stats = All_stats.replace('--', 0)

All_stats['Passing Yards'] = All_stats['Passing Yards'].astype(str).str.replace(',', '')

All_stats['Rushing Yards'] = All_stats['Rushing Yards'].astype(str).str.replace(',', '')

All_stats['Receiving Yards'] = All_stats['Receiving Yards'].astype(str).str.replace(',', '')

All_stats['Longest Pass'] = All_stats['Longest Pass'].astype(str).str.replace('T', '')

All_stats['Longest Rushing Run'] = All_stats['Longest Rushing Run'].astype(str).str.replace('T', '')

All_stats['Longest Reception'] = All_stats['Longest Reception'].astype(str).str.replace('T', '')

All_stats.loc[:, ~All_stats.columns.isin(['Player Id', 'Name', 'Position', 'Team'])] = All_stats.loc[:, ~All_stats.columns.isin(['Player Id', 'Name', 'Position', 'Team'])].astype(float)

All_stats = All_stats[All_stats['Year'] >= 1966]
#Create positional dataframes and only consider players with at least 4 qualifying seasons



QBs = All_stats[['Name', 'Year', 'Games Played', 'Passes Attempted', 'Passes Completed', 'Completion Percentage', 

                'Pass Attempts Per Game', 'Passing Yards', 'Passing Yards Per Attempt', 'Passing Yards Per Game', 

                'TD Passes', 'Percentage of TDs per Attempts', 'Ints', 'Int Rate', 'Sacks', 'Sacked Yards Lost', 'Passer Rating', 'Rushing Attempts', 

                'Rushing Attempts Per Game', 'Yards Per Carry', 'Rushing Yards Per Game', 'Rushing TDs']].copy()

QBs['TDs Per Game'] = (QBs['TD Passes'] + QBs['Rushing TDs']) / QBs['Games Played'] 

#Only consider players with more than 50 pass attempts in a season

QBs = QBs[QBs['Passes Attempted'] >= 50]

QBs = QBs.groupby('Name').filter(lambda row: row['Name'].count() > 3)

print ('# of Qualifying QBs: ' + str(len(QBs['Name'].unique())))



RBs = All_stats[['Name', 'Year', 'Games Played', 'Passes Attempted', 'Rushing Attempts', 'Rushing Attempts Per Game', 'Rushing Yards', 'Yards Per Carry', 'Rushing Yards Per Game', 'Rushing TDs',

                'Rushing First Downs', 'Percentage of Rushing First Downs', 'Fumbles', 'Receptions', 'Receiving Yards', 'Yards Per Reception', 'Yards Per Game', 

                'Receiving TDs', 'First Down Receptions']].copy()

RBs['TDs Per Game'] = (RBs['Rushing TDs'] + RBs['Receiving TDs']) / RBs['Games Played']

#Only consider players with more than 40 rush attempts and fewer than 25 pass attempts in a season

RBs = RBs[(RBs['Passes Attempted'].isnull()) | (RBs['Passes Attempted'] < 25)]

RBs = RBs[RBs['Rushing Attempts'] >= 40]

RBs = RBs.groupby('Name').filter(lambda row: row['Name'].count() > 3)

print ('# of Qualifying RBs: ' + str(len(RBs['Name'].unique())))



WRs_TEs = All_stats[['Name', 'Year', 'Games Played', 'Passes Attempted', 'Rushing Attempts', 'Rushing Attempts Per Game', 'Rushing Yards', 'Yards Per Carry', 'Rushing Yards Per Game', 'Rushing TDs',

                    'Rushing First Downs', 'Percentage of Rushing First Downs', 'Fumbles', 'Receptions', 'Receiving Yards', 'Yards Per Reception', 'Yards Per Game', 

                    'Receiving TDs', 'First Down Receptions']].copy()

WRs_TEs['TDs Per Game'] = (WRs_TEs['Rushing TDs'] + WRs_TEs['Receiving TDs']) / WRs_TEs['Games Played']

WRs_TEs['Receptions Per Game'] = WRs_TEs['Receptions'] / WRs_TEs['Games Played']

#Only consider players with at least 15 receptions and more receptions than rushes and fewer than 25 pass attempts in a season

WRs_TEs = WRs_TEs[(WRs_TEs['Passes Attempted'].isnull()) | (WRs_TEs['Passes Attempted'] < 25)]

WRs_TEs = WRs_TEs[(WRs_TEs['Receptions'] > WRs_TEs['Rushing Attempts']) & (WRs_TEs['Receptions'] >= 15)]

WRs_TEs = WRs_TEs.groupby('Name').filter(lambda row: row['Name'].count() > 3)

print ('# of Qualifying WRs/TEs: ' + str(len(WRs_TEs['Name'].unique())))

#Define functions that will be needed



#Creates list of individual dataframes for each player

def list_of_dfs(df):

	Position_names = df['Name'].unique()

	Position_dfs = []

	for name in Position_names:

		new_df = df[df['Name'] == name]

		new_df = new_df.iloc[::-1] #Put seasons in chronological order

		Position_dfs.append(new_df)

	return Position_dfs



min_max_scaler = preprocessing.MinMaxScaler()



#Normalizes data across seasons for each player 

def norm_list_of_dfs(list_of_dfs):

	norm_dfs = []

	for df in list_of_dfs:

		df_2 = df.loc[:, ~df.columns.isin(['Name', 'Year'])]

		df_norm = pd.DataFrame(min_max_scaler.fit_transform(df_2), columns = df_2.columns)

		df_norm.insert(loc=0, column = 'Name', value = df['Name'].tolist())

		df_norm.insert(loc=1, column = 'Year', value = df['Year'].tolist())

		norm_dfs.append(df_norm)

	return norm_dfs



#Finds the greatest number of seasons a player played at the position

def longest(dfs):

	max_len = 0

	for df in dfs:

		size = len(df)

		if size > max_len:

			max_len = size

	return max_len



#Fills NaN values to length of longest career so an average can be computed later

def append_nones(target_length, df):

    diff_len = target_length - len(df)

    if diff_len < 0:

        raise AttributeError('Length error list is too long.')

    return df + [np.nan] * diff_len



#Creates list of lists for the 'productivities' of every player and then averages them for the position

def avg_prod_vs_exp(list_of_dfs):

	target_length = longest(list_of_dfs)

	list_of_prods = []

	for df in list_of_dfs:

		list_of_prods.append(append_nones(target_length, df['Productivity'].tolist()))

	return np.nanmean(list_of_prods, axis=0)



#Finds average length of career among qualifying players at the position

def find_average_seasons(position_dfs):

	years_list = []

	for df in position_dfs:

		years_list.append(len(df))

	return np.mean(years_list)
#Make list of dataframes for each Position and normalize



QB_dfs = list_of_dfs(QBs)

QB_dfs_norm = norm_list_of_dfs(QB_dfs)

print ('Average # Seasons for QBs: ' + str(find_average_seasons(QB_dfs)))

#10 QBs with careers 10 years or longer, 34 12 years or longer, 13 15 years or longer



RB_dfs = list_of_dfs(RBs)

for i in range(len(RB_dfs)):

	RB_dfs[i] = RB_dfs[i].fillna(0)

RB_dfs_norm = norm_list_of_dfs(RB_dfs)

print ('Average # Seasons for RBs: ' + str(find_average_seasons(RB_dfs)))

#26 RBs with careers 10 years or longer, 8 12 years or longer, 1 15 years or longer (Marcus Allen)



WR_TE_dfs = list_of_dfs(WRs_TEs)

for i in range(len(WR_TE_dfs)):

	WR_TE_dfs[i] = WR_TE_dfs[i].fillna(0)

WR_TE_dfs_norm = norm_list_of_dfs(WR_TE_dfs)

print ('Average # Seasons for WRs/TEs: ' + str(find_average_seasons(WR_TE_dfs)))

#67 WRs/TEs with careers 10 years or longer, 31 12 years or longer, 5 15 years or longer (TO, Henry Ellard, Rice, Andre Reed, Gonzalez)
#Calculate 'Productivity' metric for each position and normalize it for each player



for df in QB_dfs_norm:

	df['Productivity'] = df['Completion Percentage'] + df['Passing Yards Per Attempt'] + df['Passing Yards Per Game'] + df['Percentage of TDs per Attempts'] + df['TDs Per Game'] - df['Int Rate'] + df['Passer Rating'] + 0.5*(df['Rushing Yards Per Game'].fillna(0))    

	prod_vals = df['Productivity'].values.reshape(-1, 1)

	prod_vals_scaled = min_max_scaler.fit_transform(prod_vals)

	df['Productivity'] = prod_vals_scaled



for df in RB_dfs_norm:

	df['Productivity'] = df['Yards Per Carry'] + df['Rushing Yards Per Game'] + df['Yards Per Reception'] + df['Yards Per Game'] + df['TDs Per Game']    

	prod_vals = df['Productivity'].values.reshape(-1, 1)

	prod_vals_scaled = min_max_scaler.fit_transform(prod_vals)

	df['Productivity'] = prod_vals_scaled

			

for df in WR_TE_dfs_norm:

	df['Productivity'] = df['Rushing Yards Per Game'] + df['Yards Per Reception'] + df['Yards Per Game'] + df['Receptions Per Game'] + df['TDs Per Game']    

	prod_vals = df['Productivity'].values.reshape(-1, 1)

	prod_vals_scaled = min_max_scaler.fit_transform(prod_vals)

	df['Productivity'] = prod_vals_scaled

    

print (QB_dfs_norm[0][['Name', 'Year', 'Productivity']])

print (RB_dfs_norm[0][['Name', 'Year', 'Productivity']])

print (WR_TE_dfs_norm[0][['Name', 'Year', 'Productivity']])
#Calculate average 'Productivity' for each position 



QBs_avg_prod_vs_exp = avg_prod_vs_exp(QB_dfs_norm)

RBs_avg_prod_vs_exp = avg_prod_vs_exp(RB_dfs_norm)

WRs_TEs_avg_prod_vs_exp = avg_prod_vs_exp(WR_TE_dfs_norm)
#Plot the results



ax = plt.subplot(111)

ax.plot(np.arange(longest(QB_dfs)), QBs_avg_prod_vs_exp, label = 'QBs')

ax.plot(np.arange(longest(RB_dfs)), RBs_avg_prod_vs_exp, label = 'RBs')

ax.plot(np.arange(longest(WR_TE_dfs)), WRs_TEs_avg_prod_vs_exp, label = 'WRs/TEs')



ax.set_xlabel('Years of Playing Experience', fontsize=20)

ax.set_ylabel('Productivity', fontsize=20)



#plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

ax.tick_params(axis='x', which='major', labelsize=16)

ax.tick_params(axis='y', which='major', labelsize=16)



plt.tight_layout()

plt.legend(prop={'size':12}, loc='lower left', frameon=False)

plt.show()