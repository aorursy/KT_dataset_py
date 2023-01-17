import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import trueskill # A python implementation of TrueSkill

import pprint # pretty printing



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# results of ~900k matches occuring prior to the other uploaded data

match_outcomes = pd.read_csv('../input/match_outcomes.csv',nrows=100000) # use 100000 matches for example



match_outcomes.head()
# I just started learning how to apply the TrueSkill rating method,

# so please mention anything that looks like it is incorrect, or will cause inaccuracy 



# there are no draws in Dota so set the probability of draw to 0

ts = trueskill.TrueSkill(draw_probability=0)

ts
# get all the account ids from match_outcomes, each ID will have a seperate rating associated with it. 

unique_acc_ids = pd.DataFrame(match_outcomes.iloc[:,1:6].unstack().unique(),

                              columns=['account_id'])

print('Number of unique account ids:', unique_acc_ids.shape[0],'\n')

print(unique_acc_ids.head())
# creating a dictionary with one default value Rating object for each account_id

rating_dict = dict()

for i in unique_acc_ids.values.ravel():

    rating_dict[i] = ts.create_rating()
len(rating_dict.keys())
# Each account_id is a key with the value being the Rating object

pprint.pprint(str(rating_dict)[:253]+'}')
def replace_anonymous(team_ids,rating_dict, team_name, ts_obj):

    """Creates fake ratings for players without account_id. 

    Uses the mean of other players skill on team to create fake ratings.

    

    :param team_ids: account_ids for one team(5 players)

    :param rating_dict: dictionary with account_id for keys, and trueskill Rating obj for values

    :param team_name: either 'radiant' or 'dire', used to name pseudo players

    :param ts_obj: trueskill object from the trueskill python library   

    :return: a dictionary with account_ids, and rating objects for the team

    """

    

    team_dict = dict()

    non_anon = []

    

    sum_mu = 0

    sum_sigma = 0

    

    # get the sum of non-anon players mu, and sigma

    non_anons = 0

    for i in team_ids:

        if i != 0: # 0 is the account_id for anons 

            sum_mu += rating_dict[i].mu

            sum_sigma += rating_dict[i].sigma

            non_anons += 1

    

    # for the case in which all players on the team are anonymous

    if non_anons == 0:

        for e in range(5):

            team_dict[team_name + str(e)] = ts_obj.create_rating()

        return team_dict

        

    # take the mean of the mean and stdev

    # note what other methods could be used here?

    mean_mu = sum_mu/non_anons

    mean_sigma = sum_sigma/non_anons



    # if a player has an account_id add them to the team dict otherwise add an fake player

    for e,i in enumerate(team_ids):

        

        if i == 0: 

            team_dict[team_name + str(e)] = ts_obj.create_rating(mean_mu, mean_sigma)

        else:

            team_dict[i] = rating_dict[i]

        

    return team_dict



def update_ratings(new_ratings, rating_dict):

    """Updates the rating dictionary 

    Note assumes that pseudo players have string keys, in new_ratings, and real players have non-string keys 

    

    :param new_ratings: a dictionary of ratings, keys are account_id, values are trueskill Rating object

    :param rating_dict: the rating dictionary being used to keep track of player ratings

    :return: the updated rating dictionary 

    """

    for key in new_ratings.keys():

        if type(key) is not str:

            rating_dict[key] = new_ratings[key]

    return rating_dict
match_groups = match_outcomes.groupby('match_id')



# just to see what this looks like

for e,group in enumerate(match_groups):

    break

print('The group key(match_id):',group[0],'\n')

print('The group data with dire on top row and radiant on bottom.\n',group[1])
%%time 



# iterate through the matches updating player ratings after each match. 



for e,group in enumerate(match_groups):

    # This assumes that radiant is always on the second row, if you shuffle the table the assumption may not hold

    # additionaly any modification to match_outcomes will break this. But its a little faster this way;)

    radiant_ids = group[1].iloc[1,1:6] 

    dire_ids = group[1].iloc[0,1:6]

    

    radiant_dict = replace_anonymous(radiant_ids, rating_dict, 'radiant', ts)

    dire_dict = replace_anonymous(dire_ids, rating_dict, 'dire', ts)

    

    if group[1].iloc[1,8] == 1: # radiant won

        updated_radiant, updated_dire = ts.rate([radiant_dict, dire_dict], ranks=[0,1]) # for ranks 0 is winner 

    else: # dire won

        updated_radiant, updated_dire = ts.rate([radiant_dict, dire_dict], ranks=[1,0])

    

    # update the rating dictionary

    rating_dict = update_ratings(updated_radiant, rating_dict)

    rating_dict = update_ratings(updated_radiant, rating_dict)
# transform the updated rating dictionary into a pandas DataFrame

rating_arr = np.zeros((len(rating_dict.keys()), 3))



for e,i in enumerate(rating_dict.keys()):

    rating_arr[e,0] = i

    rating_arr[e,1] = rating_dict[i].mu

    rating_arr[e,2] = rating_dict[i].sigma
rating_df = pd.DataFrame(rating_arr, columns=['acccount_id', 'trueskill_mu','trueskill_sigma'])

rating_df.head(10)
# player win counts and trueskill scores

player_ratings = pd.read_csv('../input/player_ratings.csv')

player_ratings.head()
player_ratings.shape
frequent_players = player_ratings.query('total_matches >= 50')

frequent_players.shape
frequent_players = frequent_players.sort_values(by='total_matches')
frequent_players.head(10)
frequent_players.tail(10)
players = pd.read_csv('../input/players.csv')
# join ratings with the players table

players = pd.merge(players, player_ratings, how='left', left_on='account_id', right_on='account_id')

frequent_players =  players.query('total_matches >= 50').copy()



# some of these variables don't have order, and I don't want to plot them against skill

to_drop = ['match_id', 'hero_id','player_slot', 

           'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'item_5', 

           'leaver_status', 'unit_order_none', 'stuns','total_matches',

           'total_wins', 'trueskill_sigma']



frequent_players.drop(to_drop, axis=1, inplace=True)



frequent_players.shape
# get rid a few rows that are all nan or almost all nan

frequent_players  = frequent_players.loc[:,frequent_players.isnull().sum() < 240000]

frequent_players.shape
frequent_players.columns
fp_groups = frequent_players.groupby('account_id')

means = fp_groups.mean() 
# this will do regression plots for all of the above columns against skill but it doesn't \

# run on Kernels in a reasonable amount of time. 



# col = frequent_players.columns

# plot_col = []

# for e,i in enumerate(col):

#    if e != 0 and i != 'trueskill_mu': 

#        plot_col.append(i)

#    if e%3 == 0 and e!=0:

#        with sns.plotting_context("notebook", font_scale=0.65):

#            sns.pairplot(means, kind='reg', y_vars=['trueskill_mu'], x_vars=plot_col)

#        

#        plot_col=[]

#        break

        
# Some player actions seem to also be associated with different skill levels

plot_col = ['unit_order_cast_no_target', 'unit_order_drop_item', 'unit_order_move_item']

with sns.plotting_context("notebook", font_scale=0.8):

    sns.pairplot(means, kind='reg', y_vars=['trueskill_mu'], x_vars=plot_col)
# take a look at some regular stats in relation to the trueskill mu

plot_col = ['xp_per_min', 'gold_per_min', 'deaths']

with sns.plotting_context("notebook", font_scale=0.7):

    sns.pairplot(means, kind='reg', y_vars=['trueskill_mu'], x_vars=plot_col)