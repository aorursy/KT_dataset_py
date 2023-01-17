# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
match_df = pd.read_csv("/kaggle/input/ufc-fights-2010-2020-with-betting-odds/data.csv")

#Let's put all the labels in a dataframe

match_df['label'] = ''

#If the winner is not Red or Blue we can remove it.

mask = match_df['Winner'] == 'Red'

match_df['label'][mask] = 0

mask = match_df['Winner'] == 'Blue'

match_df['label'][mask] = 1



#df["Winner"] = df["Winner"].astype('category')

match_df = match_df[(match_df['Winner'] == 'Blue') | (match_df['Winner'] == 'Red')]





#Make sure label is numeric

match_df['label'] = pd.to_numeric(match_df['label'], errors='coerce')



#Let's fix the date

match_df['date'] = pd.to_datetime(match_df['date'])
rankings_df = pd.read_csv("/kaggle/input/ufc-rankings/rankings_history.csv")

rankings_df['date'] = pd.to_datetime(rankings_df['date'])
weightclass_list = rankings_df.weightclass.unique()

print(weightclass_list)
print(rankings_df.columns)
date_list = rankings_df.date.unique()

display(date_list)
print(min(date_list))
display(rankings_df.head())
display(match_df.columns)
#fighter_name: the fighter name

#date: the date of the event

#wc: the weightclass where we are looking for a rank.



def return_rank(fighter_name, date, wc):

    #If we can't find the rank we are going to return an empty string

    rank = ''

    

    #We can stop cycling through dates once we have moved past the event

    keep_going = True;

    

    #Once we move past the event we are going to have to look at the previous list of rankings

    previous_d = ''

    

    for d in date_list:

        if keep_going:

            time_dif =  (d - date).total_seconds()

            #If time_dif is not negative we can stop cycling

            if time_dif > -1:

                keep_going = False

                #print(fighter_name, time_dif, date, wc, d)

                temp_rankings_df = rankings_df[rankings_df['date']==previous_d].copy()

                temp_rankings_df = temp_rankings_df[temp_rankings_df['weightclass']==wc]

                temp_rankings_df = temp_rankings_df[temp_rankings_df['fighter']==fighter_name]

                #This means we have a match.  We need to return the rank

                if len(temp_rankings_df) > 0:

                    rank = int(temp_rankings_df.iloc[0]['rank'])

                    #display(rank)

                    #print(fighter_name)

                #print(len(temp_rankings_df))

            else:

                previous_d = d

    if isinstance(rank, int):

        #print(rank)

        return(rank)

    else:

        return('')
match_df['B_match_weightclass_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         x['weight_class']),axis=1)

match_df['R_match_weightclass_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         x['weight_class']),axis=1)
match_df['R_Women\'s Flyweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Flyweight'),axis=1)
match_df['R_Women\'s Featherweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Featherweight'),axis=1)
match_df['R_Women\'s Strawweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Strawweight'),axis=1)
match_df['R_Women\'s Bantamweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Bantamweight'),axis=1)
match_df['R_Heavyweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Heavyweight'),axis=1)
match_df['R_Light Heavyweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Light Heavyweight'),axis=1)
match_df['R_Middleweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Middleweight'),axis=1)
match_df['R_Welterweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Welterweight'),axis=1)
match_df['R_Lightweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Lightweight'),axis=1)
match_df['R_Featherweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Featherweight'),axis=1)
match_df['R_Bantamweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Bantamweight'),axis=1)
match_df['R_Flyweight_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Flyweight'),axis=1)
match_df['R_Pound-for-Pound_rank'] = match_df.apply(lambda x: return_rank(x['R_fighter'],

                                                                         x['date'],

                                                                         'Pound-for-Pound'),axis=1)
match_df['B_Women\'s Flyweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Flyweight'),axis=1)
match_df['B_Women\'s Featherweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Featherweight'),axis=1)
match_df['B_Women\'s Strawweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Strawweight'),axis=1)
match_df['B_Women\'s Bantamweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Women\'s Bantamweight'),axis=1)
match_df['B_Heavyweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Heavyweight'),axis=1)
match_df['B_Light Heavyweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Light Heavyweight'),axis=1)
match_df['B_Middleweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Middleweight'),axis=1)
match_df['B_Welterweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Welterweight'),axis=1)
match_df['B_Lightweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Lightweight'),axis=1)
match_df['B_Featherweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Featherweight'),axis=1)
match_df['B_Bantamweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Bantamweight'),axis=1)
match_df['B_Flyweight_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Flyweight'),axis=1)
match_df['B_Pound-for-Pound_rank'] = match_df.apply(lambda x: return_rank(x['B_fighter'],

                                                                         x['date'],

                                                                         'Pound-for-Pound'),axis=1)
def return_better_rank(r_rank, b_rank):

    if (r_rank == ''):

        if b_rank != '':

            return('Blue')

        else:

            return('neither')

    if (b_rank == ''):

        return('Red')

    r_rank = int(r_rank)

    b_rank = int(b_rank)

    if (r_rank < b_rank):

        return('Red')

    else:

        return('Blue')

    return('neither')
match_df['better_rank'] = match_df.apply(lambda x: return_better_rank(x['R_match_weightclass_rank'],

                                                                         x['B_match_weightclass_rank']),axis=1)
display(match_df.head())
display(match_df.columns)
temp_df = match_df[match_df['better_rank']=='Red'].copy()

red_favorite_count = (len(temp_df))

temp_df = temp_df[temp_df['Winner']=='Red']

red_winner_count = len(temp_df)



red_pct = (red_winner_count / red_favorite_count)



temp_df = match_df[match_df['better_rank']=='Blue'].copy()

blue_favorite_count = (len(temp_df))

temp_df = temp_df[temp_df['Winner']=='Blue']

blue_winner_count = len(temp_df)



blue_pct = (blue_winner_count / blue_favorite_count)

print('When Red has the better rank they win ', "{:.2f}".format(red_pct*100), '% of the time')

print('When Blue has the better rank they win ', "{:.2f}".format(blue_pct*100), '% of the time')