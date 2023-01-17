# Import all the necessary python libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.options.mode.chained_assignment = None    # To avoid the SettingWithCopyWarning



# Import the datasets

deliveries_df = pd.read_csv('../input/deliveries.csv')
deliveries_df.head()
deliveries_df.columns
# We do not need all the columns for our present analysis. 

# So let us cut down the dataframe a little to make it easier to work with



column_heads = ["match_id", "batsman", "batsman_runs"]



deliveries_batsmen = deliveries_df[column_heads]



deliveries_batsmen.sample()
# Adding columns with 4s and 6s encoded

deliveries_batsmen["Fours"] = np.where(deliveries_batsmen["batsman_runs"] == 4, 1, 0)

deliveries_batsmen["Sixes"] = np.where(deliveries_batsmen["batsman_runs"] == 6, 1, 0)



deliveries_batsmen
batsmen_info = deliveries_batsmen.groupby(['batsman']).agg({'match_id': lambda x: x.nunique(), 'batsman_runs': 'sum', 'Fours': 'sum', 'Sixes': 'sum'}).reset_index()



# Here, I am grouping by batsman. 

# So all the occurences of a particular batsman will be grouped together.

# I am then using aggregations to compute on the individual columns.

# For number of matches played, I am counting the number of unique 'match_id' 

# for a particular group (batsman).

# For total runs scored, I am summing up the 'batsman_runs' for a particular group (batsman).

# For total 4s and 6s, adding up the valus in their respective columns, since they have been 

# already encoded as 1 and 0.



batsmen_info.head()
# Data such as highscore and number of 30s is dependent on each match and thus we need to groupby 'match_id' too.



x = deliveries_batsmen.groupby(['match_id', 'batsman'])



# Keep in mind that 'x' is a DataFrameGroupBy object and not DataFrame object.



# Calculate runs scored in each match and then find max of runs scored in each match to get highscore.

p = x.sum().reset_index().groupby('batsman').agg({'batsman_runs': 'max'}).reset_index()



# Calculate runs scored in each match, encode scores compared to 30s and calculate number of 30s.

q = x.sum().reset_index().groupby(['match_id', 'batsman']).agg(

    {'batsman_runs': lambda x: 1 if (x > 29).any() else 0}).reset_index().groupby('batsman').agg({'batsman_runs': 'sum'}).reset_index()



# Calculate number of occurences of batsmen for every match and then calculate balls faced. 

# We will need this for strike rate.

r = x.count().reset_index().groupby('batsman').agg({'batsman_runs': 'sum'}).reset_index()



# Rename columns

q = q.rename(columns={'batsman_runs': "30s"})

p = p.rename(columns={'batsman_runs': "HS"})

r = r.rename(columns={'batsman_runs': "Ball faced"})



# Merge into main dataframe

batsmen_info = pd.merge(batsmen_info, p, on=['batsman'])

batsmen_info = pd.merge(batsmen_info, q, on=['batsman'])

batsmen_info = pd.merge(batsmen_info, r, on=['batsman'])



batsmen_info.head()
# Average and strike rate

batsmen_info['Average'] = np.around(batsmen_info['batsman_runs'] / batsmen_info['match_id'], 2)

batsmen_info['SR'] = np.around((batsmen_info['batsman_runs'] / batsmen_info['Ball faced']) * 100, 2)



batsmen_info.head()
# Reorder/rename columns for presentation

batsmen_info = batsmen_info.rename(columns={'batsman': 'Batsman', 'batsman_runs': 'Runs', 'match_id': 'Matches'})

sequence = ['Batsman', 'Matches', 'Runs', 'Average', 'SR', '30s', 'HS', 'Sixes', 'Fours']

batsmen_info = batsmen_info[sequence]



batsmen_info.nlargest(10, 'Runs').reset_index(drop='T')
# We will be using the same deliveries.csv as before.



# Lets take a look at the columns

deliveries_df.columns
# To recreate our information panel, we do not need all the columns

column_heads = list(set(list(deliveries_df.columns)).difference(set(['inning', 'non_striker', 'is_super_over', 'legbye_runs', 'bye_runs'])))



column_heads
# Taking a look at the sub-dataframe

deliveries_bowlers = deliveries_df[column_heads]



deliveries_bowlers.head()
# Getting number of unique bowlers. This should tell us how many bowlers should appear in our final dataframe.

deliveries_bowlers['bowler'].nunique()
# There are a lot of Nan values, which I replace by 0

deliveries_bowlers.fillna("0", inplace=True)



# Adding columns for 4s and 6s

deliveries_bowlers["Fours"] = np.where(deliveries_bowlers["batsman_runs"] == 4, 1, 0)

deliveries_bowlers["Sixes"] = np.where(deliveries_bowlers["batsman_runs"] == 6, 1, 0)



# Determining whether wicket which fell will be attribute to bowler



# Looking at the different types of dismissals

deliveries_bowlers['dismissal_kind'].unique()
# Out of these we need only a few specific ones to qualify as wicket taken by bowler

dismissal_kind = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']

deliveries_bowlers["Wickets"] = np.where(deliveries_bowlers['dismissal_kind'].isin(dismissal_kind), 1, 0)



deliveries_bowlers.sample()
# Let us now look at bowler performance in every match

bowlers_info = deliveries_bowlers.groupby(['bowler', 'match_id'], as_index=False).agg({'over': lambda x: x.nunique(), 'total_runs': 'sum', 'Fours': 'sum', 'Sixes': 'sum', 'Wickets': 'sum'}).reset_index()



bowlers_info.sample()
# Encoding for number of 3 wickets taken in a match

bowlers_info['3WI'] = np.where(bowlers_info['Wickets']>2, 1, 0)



# Building career stats dataframe

career_stats = bowlers_info.groupby('bowler', as_index=False).agg({'match_id': 'count', 'total_runs': 'sum', 'over': 'sum', 'Fours': 'sum', 'Sixes': 'sum', 'Wickets': 'sum', '3WI': 'sum'}).reset_index()



# Calculating best bowling figures in a match

most_wickets_in_match = bowlers_info.groupby(['bowler']).agg({'Wickets': 'max'}).reset_index()

bowlers_info = pd.merge(bowlers_info, most_wickets_in_match, on=['bowler', 'Wickets'])



least_runs_for_most_wickets = bowlers_info.groupby(['bowler']).agg({'total_runs': 'min'}).reset_index()

bowlers_info = pd.merge(bowlers_info, least_runs_for_most_wickets, on=['bowler', 'total_runs'])



bowlers_info.head()
# Best bowling figures in presentable format

bowlers_info['BB'] = (bowlers_info['Wickets'].map(str)) + '/' + (bowlers_info['total_runs'].map(str))



bowlers_info.head()
# Looking at number of bowlers

bowlers_info['bowler'].count()
# This means that there is one bowler who has taken the same number of highest wickets in a match and given away the same number of runs twice in his career.



# Looking at first occurence

bowlers_info[bowlers_info.duplicated(['bowler'], keep='first')]
# Looking at second occurence

bowlers_info[bowlers_info.duplicated(['bowler'], keep='last')]
# We will keep the one where he was not hit for a six

bowlers_info = bowlers_info.drop_duplicates('bowler', keep='last')



# Merging with career stats dataframe

career_stats = pd.merge(career_stats, bowlers_info[['bowler', 'BB']], on='bowler')



career_stats.head()
# Renaming columns and adding new ones

career_stats = career_stats.rename(columns={'bowler': 'Bowler', 'total_runs': 'Runs', 'match_id': 'Matches', 'over': 'Overs'})

career_stats['Average'] = np.around(career_stats['Runs'] / career_stats['Wickets'], 2)

career_stats['Economy'] = np.around(career_stats['Runs'] / career_stats['Overs'], 2)

career_stats['SR'] = np.around((career_stats['Overs']*6) / career_stats['Wickets'], 2)



# Reordering columns for presentation

sequence = ['Bowler', 'Matches', 'Wickets', 'Average', 'Economy', 'SR', 'BB',  '3WI', 'Fours', 'Sixes']



career_stats = career_stats[sequence]



career_stats.sort_values(by='Wickets', ascending=False).reset_index(drop='T')[:10]



# For some reason, nlargest() was buggy and giving me some identical repeated rows here on Kaggle (worked fine on my local machine). Consequently, I had t go with sort_values