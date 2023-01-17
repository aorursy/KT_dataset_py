import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



cols = [

    'tourney_id', # Id of Tournament

    'tourney_name', # Name of the Tournament

    'surface', # Surface of the Court (Hard, Clay, Grass)

    'draw_size', # Number of people in the tournament

    'tourney_level', # Level of the tournament (A=ATP Tour, D=Davis Cup, G=Grand Slam, M=Masters)

    'tourney_date', # Start date of tournament

    'match_num', # Match number

    'winner_id', # Id of winner

    'winner_seed', # Seed of winner

    'winner_entry', # How the winner entered the tournament

    'winner_name', # Name of winner

    'winner_hand', # Dominant hand of winner (L=Left, R=Right, U=Unknown?)

    'winner_ht', # Height in cm of winner

    'winner_ioc', # Country of winner

    'winner_age', # Age of winner

    'winner_rank', # Rank of winner

    'winner_rank_points', # Rank points of winner

    'loser_id',

    'loser_seed',

    'loser_entry',

    'loser_name',

    'loser_hand',

    'loser_ht',

    'loser_ioc',

    'loser_age',

    'loser_rank',

    'loser_rank_points',

    'score', # Score

    'best_of', # Best of X number of sets

    'round', # Round

    'minutes', # Match length in minutes

    'w_ace', # Number of aces for winner

    'w_df', # Number of double faults for winner

    'w_svpt', # Number of service points played by winner

    'w_1stIn', # Number of first serves in for winner

    'w_1stWon', # Number of first serve points won for winner

    'w_2ndWon', # Number of second serve points won for winner

    'w_SvGms', # Number of service games played by winner

    'w_bpSaved', # Number of break points saved by winner

    'w_bpFaced', # Number of break points faced by winner

    'l_ace',

    'l_df',

    'l_svpt',

    'l_1stIn',

    'l_1stWon',

    'l_2ndWon',

    'l_SvGms',

    'l_bpSaved',

    'l_bpFaced'

]



df = pd.concat([

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2000.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2001.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2002.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2003.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2004.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2005.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2006.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2007.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2008.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2009.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2010.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2011.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2012.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2013.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2014.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2015.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2016.csv', usecols=cols),

    pd.read_csv('/kaggle/input/atp-matches-dataset/atp_matches_2017.csv', usecols=cols),

],ignore_index=True) #have to make sure that the index will not be duplicated



df.tail()
roger = df.loc[(df['winner_name'] == 'Roger Federer') | (df['loser_name'] == 'Roger Federer')].copy()

#we want to analyze his performance over time but date is not float 64, so let's change it to datettime datatype.

roger.tourney_date.apply(lambda x: '%.0f' % round(x,0))

roger.loc[:,'tourney_date'] = pd.to_datetime(roger['tourney_date'], format='%Y%m%d')
# Let's look at his serve performance over the time. However, we have to look at the number of serve whether it is winner or loser based on loser or winner

# how important is the serve

# let's define variable that we will use a lot

rogerwin = roger.loc[roger['winner_name'] == 'Roger Federer'].copy()

rogerloss = roger.loc[roger['loser_name'] == 'Roger Federer'].copy()

# print(f'Number of wins: {rogerwin.count()[0]}')

# print(f'Number of losses: {rogerloss.count()[0]}')

    

    #let's plot the number of wins and losses from 2000 to 

fig = plt.figure(figsize=(5,5)) 

import matplotlib as mpl



plt.title("No. of Roger's Wins and Losses throughout his careers")

plt.pie([rogerwin.count()[0],rogerloss.count()[0]],labels = [f'{rogerwin.count()[0]} matches won',f'{rogerloss.count()[0]} matches lost'],textprops={'fontsize': 20})
years = []

for i in range(0,18):

    years.append(i+2000)



annualwin = rogerwin.groupby(rogerwin.tourney_date.dt.year).count().tourney_id

annualloss = rogerloss.groupby(rogerloss.tourney_date.dt.year).count().tourney_id



plt.xticks(np.arange(2000,2017,2), rotation=45)

plt.plot(years,annualwin,label='Win')

plt.plot(years[:-1],annualloss,label='Loss')

plt.legend()

# plt.subplot(2,2,1)

# plt.bar(years[:],annualwin)



# plt.subplot(2,2,2)

# plt.bar(years[:-1],annualloss)

# Group by the tour_id where level == Grandslam

# However,l this is until 2017

tour = rogerwin.loc[rogerwin.tourney_level == 'G'].groupby(rogerwin.tourney_id).count()

championship = tour.loc[tour.tourney_id == 7]

plt.title('No. of Grandslam Championships')

plt.yticks([1,2,3])

grandslams = championship.groupby(championship.index.map(lambda x: x[0:4])).count()

plt.bar(grandslams.index,grandslams.tourney_id)



# Will do this later



# let's look at whether he has finished a match with fewer number of matches or not the time by looking at the number of sets overall, only when he won.

# On the other hand, did he lose more difficult? meaning does it require more sets for opponents to win over him?

# This is difficult because for grandslam, you need 3 sets to win while you only need two for mormal matches

rogerwin['No. of sets'] = rogerwin.score.map(lambda x: len(x.split(' ')) if (len(x.split(' ')) <= 5) else 5)

roger.head()

# roger.score.str.split(' ')

# roger.drop('No. of sets', axis = 1)





# average game when roger win through each year

# this number is kinda difficult to see because grand slams require 3 sets while normal tournament would be 2

# averageNoSets = roger.loc[roger.winner_name == 'Roger Federer'].groupby(roger.tourney_date.dt.year).mean()['No. of sets']

averageNoSets = roger.loc[roger.winner_name == 'Roger Federer'].groupby(roger.tourney_date.dt.year).mean()

averageNoSets.head()

# plt.plot(years,averageNoSets)







#let's look at his serve performance by

# 1) number of his aces/servepoints, average percentage of ace per match!

# 2) number of his double faults/servepoints

# 3) 1stin = 1st serve in



def get_aces_percent(row):

    if row['winner_name'] == 'Roger Federer':

        val = row.w_ace/row.w_svpt

    elif row['loser_name'] == 'Roger Federer':

        val = row.l_ace/row.l_svpt

    return val



def get_double_faults_percent(row):

    if row['winner_name'] == 'Roger Federer':

        val = row.w_df/row.w_svpt

    elif row['loser_name'] == 'Roger Federer':

        val = row.l_df/row.l_svpt

    return val



def get_1st_serve_in_percent(row):

    if row['winner_name'] == 'Roger Federer':

        val = row.w_1stIn/row.w_svpt

    elif row['loser_name'] == 'Roger Federer':

        val = row.l_1stIn/row.l_svpt

    return val



def get_first_serve_win_percent(row):

    if row['winner_name'] == 'Roger Federer':

        val = row.w_1stWon/row.w_svpt

    elif row['loser_name'] == 'Roger Federer':

        val = row.l_1stWon/row.l_svpt

    return val



def get_second_serve_percent(row):

    if row['winner_name'] == 'Roger Federer':

        val = row.w_2ndWon/row.w_svpt

    elif row['loser_name'] == 'Roger Federer':

        val = row.l_2ndWon/row.l_svpt

    return val







roger['roger_aces_percentage'] = roger.apply(get_aces_percent, axis=1)

roger['roger_double_faults_percentage'] = roger.apply(get_double_faults_percent, axis=1)

roger['roger_1st_serve_in_percent'] = roger.apply(get_1st_serve_in_percent, axis=1)

roger['roger_first_serve_win_percentage'] = roger.apply(get_first_serve_win_percent, axis=1)

roger['roger_second_serve_win_percentage'] = roger.apply(get_second_serve_percent, axis=1)



#group by year

groupbyyear = roger.groupby(roger.tourney_date.dt.year).mean()





fig = plt.figure()

fig, (ax1,ax2) = plt.subplots(

    nrows=2,

    ncols=1,

    sharex=True

)



fig.set_size_inches(15,10)

fig.subplots_adjust(wspace=0.5)



ax1.set_title('Percentage of Aces and Double Faults')



ax1.set_ylabel('percentage')

ax1.set_xticks(np.arange(2000,2017,2))

ax1.plot(groupbyyear.index,groupbyyear['roger_aces_percentage'], label='Ace')

ax1.plot(groupbyyear.index,groupbyyear['roger_double_faults_percentage'],label='Double Faults')

ax1.legend()



ax2.plot(groupbyyear.index,groupbyyear['roger_1st_serve_in_percent'])

ax2.set_ylabel('percentage')

ax2.set_title('Percentage of First Serve In')

ax2.set_xlabel('year')



# groupbyyear = roger.groupby(roger.tourney_date.dt.year).sum()

# groupbyyear

# Now I jsut realized that I should make a new variable that has all the info when Roger won and when he lost.

roger_win = roger.loc[roger.winner_name == 'Roger Federer', :]

roger_loss = roger.loc[roger.loser_name == 'Roger Federer', :]
def get_breakpoint_saved_percent(row):

    if row['winner_name'] == 'Roger Federer':

        if row.w_bpFaced != 0:

            val = row.w_bpSaved/row.w_bpFaced

        else:

            val = 0

    elif row['loser_name'] == 'Roger Federer':

        if row.l_bpFaced != 0:

            val = row.l_bpSaved/row.l_bpFaced

        else:

            val = 0

    return val



roger['roger_breakpoint_saved_percent'] = roger.apply(get_breakpoint_saved_percent, axis=1)



mental_level = roger.groupby(roger['tourney_date'].dt.year).mean().roger_breakpoint_saved_percent

mental_level





plt.title("Roger' Saved break Points")

plt.xlabel('years')

plt.ylabel('percentage')

plt.xticks(np.arange(2000,2017,2))

# plt.plot(np.arange(2000,2018),mental_level)

plt.plot(years,mental_level)
loss = roger.loc[roger.winner_name != 'Roger Federer', ['winner_name']]

# loss.rename(r={'winner_name': 'name'})

loss.columns = ['name']

loss['status'] = 'loss'

win = roger.loc[roger.winner_name == 'Roger Federer', ['loser_name']]

win.columns = ['name']

win['status'] = 'win'

opponents = pd.concat([win,loss])

# opponents

opponents.groupby('name').count().sort_values('status', ascending = False)
numberofmatches = opponents.groupby('name').count().sort_values('status', ascending = False)



numberoflosses = opponents.loc[opponents.status == 'loss', :].groupby('name').count().sort_values('status', ascending = False)



fig = plt.figure(figsize=(10,5))



plt.xticks(rotation=45)

plt.xlabel('opponents')

plt.ylabel('Number of Matches')





plt.bar(numberofmatches.index[0:10],numberofmatches.status[0:10], label='win')

plt.bar(numberoflosses.index[0:10],numberoflosses.status[0:10], label='loss')

plt.legend()

plt.title("Roger's head-to-head against his opponents")
opponents_grouped = pd.DataFrame()

opponents_grouped['No. of matches'] = numberofmatches.status

opponents_grouped['No. of winning'] = opponents.loc[opponents.status=='win'].groupby('name').count().status

opponents_grouped['percentage'] = opponents_grouped['No. of winning']/opponents_grouped['No. of matches']

opponents_grouped.loc[opponents_grouped['No. of matches'] > 10].sort_values('percentage',ascending = True).head()
opponents.loc[opponents.status=='win'].groupby('name').count()

plt.xlabel('opponent name')

plt.ylabel("Percentage of Roger's winning")

plt.xticks(rotation='vertical')

# plt.bar(opponents_grouped.sort_values('percentage',ascending = True).index[0:10], opponents_grouped.sort_values('percentage',ascending = True).percentage[0:10])



plt.bar(opponents_grouped.loc[opponents_grouped['No. of matches'] > 10].sort_values('percentage',ascending = True).index[0:10], opponents_grouped.sort_values('percentage',ascending = True).percentage[0:10])

plt.title("Roger's Strongest Opponents")
# # let see the number of matches roger participated in each year and the number of matches he won!

# # Number of matches roger participated

# numberOfMatchEachYear = roger.groupby(roger.tourney_date.dt.year).count()['tourney_id']



# rogerWin = roger.loc[roger.winner_name == 'Roger Federer',:].groupby(roger.tourney_date.dt.year).count()['tourney_id']

# rogerLoss = roger.loc[roger.loser_name == 'Roger Federer',:].groupby(roger.tourney_date.dt.year).count()['tourney_id']



# fig = plt.figure()

# fig, ax = plt.subplots(

#     nrows=2,

#     ncols=1

# )



# fig.set_size_inches(15,10)

# fig.subplots_adjust(wspace=0.5)





# plt.ylabel('number of matches participated')

# plt.xticks(np.arange(2000,2017,2))

# plt.ttle('Number of matches')

# plt.bar(years, numberOfMatchEachYear)

# plt.bar(years, rogerWin)










