import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

%matplotlib inline
matches = pd.read_csv("../input/matches.csv")

matches.head()
deliveries = pd.read_csv("../input/deliveries.csv")

deliveries.head()
batsmen_score = pd.DataFrame(deliveries.groupby(['match_id','batsman']).sum()['batsman_runs'])

batsmen_score.head()
plt.rcParams['figure.figsize'] = 10,5

batsmen_score.plot(kind = 'hist',fontsize = 20)

plt.xlabel('Runs Scored',fontsize = 20)

plt.ylabel('Number of Times',fontsize = 20)

plt.title('Histogram for Runs Scored',fontsize = 20)

plt.show()
batsmen_score_20 = pd.DataFrame(deliveries.groupby(['match_id','batsman']).agg({'batsman_runs' : 'sum', 'ball' :'count'}))

batsmen_score_20[batsmen_score_20['ball'] > 20].plot(kind = 'scatter', x = 'ball',y = 'batsman_runs')

plt.xlabel('Ball Faced',fontsize = 20)

plt.ylabel('Runs Scored',fontsize = 20)

plt.title('Runs Scored vs Balls Faced',fontsize = 20)

plt.show()
batsmen_strikerate = batsmen_score_20[batsmen_score_20['ball'] >= 15]

batsmen_strikerate['Strike Rate'] = batsmen_strikerate['batsman_runs']/batsmen_strikerate['ball']*100

batsmen_strikerate.head()
ax = batsmen_strikerate.plot(kind = 'scatter',x = 'batsman_runs', y = 'Strike Rate')

plt.xlabel('Runs Scored',fontsize = 20)

plt.ylabel('Strike Rate',fontsize = 20)

plt.title('Innings Progression with Runs',fontsize = 20)

plt.show()
ax = batsmen_strikerate.plot(kind = 'scatter', x ='ball',y = 'Strike Rate',color = 'y')

batsmen_strikerate.groupby(['ball']).max().plot(kind = 'line',y = 'Strike Rate',ax = ax,color = 'green',label = 'Max Strike Rate')

batsmen_strikerate.groupby(['ball']).min().plot(kind = 'line',y = 'Strike Rate',ax = ax,color = 'red',label = 'Min Strike Rate')

plt.xlabel('Ball Faced',fontsize = 20)

plt.ylabel('Strike Rate',fontsize = 20)

plt.title('Strike Rate Progression with Balls',fontsize = 20)

plt.show()
batsmen_strikerate.boxplot(column = ['Strike Rate'])
batsmen_strikerate[batsmen_strikerate['ball'] > 30].sort_values(by = 'Strike Rate',ascending = False).head()

#Balls Faced greated than 30
batsmen_strikerate[batsmen_strikerate['ball'] > 40].sort_values(by = 'Strike Rate',ascending = False).head()

#Balls Faced greated than 40
batsmen_strikerate[batsmen_strikerate['ball'] > 50].sort_values(by = 'Strike Rate',ascending = False).head()

#Balls Faced greated than 50
batsmen_strikerate[batsmen_strikerate['ball'] > 60].sort_values(by = 'Strike Rate',ascending = False).head()

#Balls Faced greated than 60
batsmen_strikerate[batsmen_strikerate['ball'] > 70].sort_values(by = 'Strike Rate',ascending = False).head()

#Balls Faced greated than 70
aggregatedata = pd.merge(matches,deliveries, left_on = 'id',right_on = 'match_id')

aggregatedata.columns
batsmen_strikerate_season = pd.DataFrame(deliveries.groupby(['batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))

batsmen_strikerate_season['Strike Rate'] = batsmen_strikerate_season['batsman_runs']/batsmen_strikerate_season['ball']*100

batsmen_strikerate_season = batsmen_strikerate_season.sort_values(by ='Strike Rate' , ascending = False)

batsmen_strikerate_season[batsmen_strikerate_season['batsman_runs'] > 2500] 

# We have taken runs greater then 2500 So that we take a significant amount of runs
colors = cm.rainbow(np.linspace(0,1,len(batsmen_strikerate_season[batsmen_strikerate_season['batsman_runs'] > 2500])))

batsmen_strikerate_season[batsmen_strikerate_season['batsman_runs'] > 2500].plot(kind = 'bar',y = 'Strike Rate',

                                                                                color = colors,legend = '',fontsize = 10)

plt.xlabel('Batsman Name',fontsize = 20)

plt.ylabel('Strike Rate',fontsize = 20)

plt.show()
batsmen_strikerate_eseason = pd.DataFrame(aggregatedata.groupby(['season','batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))

batsmen_strikerate_eseason['Strike Rate'] = batsmen_strikerate_eseason['batsman_runs']/batsmen_strikerate_eseason['ball']*100

batsmen_strikerate_eseason = batsmen_strikerate_eseason.sort_values(by =['season','Strike Rate'] , ascending = False)

batsmen_strikerate_eseason.reset_index(inplace = True)

batsmen_strikerate_eseason[batsmen_strikerate_eseason['batsman_runs'] > 300].head()



# We have taken runs greater then 300 So that we take a significant amount of runs
colors = cm.rainbow(np.linspace(0,1,10))

plt.rcParams['figure.figsize'] = 10,5

for title,group in batsmen_strikerate_eseason.groupby('season'):

    group[group['batsman_runs'] > 300].head(10).plot(x = 'batsman',y = 'Strike Rate',kind = 'bar',legend = '',

                                                     color = colors,fontsize = 10)

    plt.xlabel('Batsman Name',fontsize = 20)

    plt.ylabel('Strike Rate',fontsize = 20)

    plt.title('Top 10 Strike Rate in Season %s '%title,fontsize = 20)

plt.show()
strikerate_inning = pd.DataFrame(deliveries.groupby(['match_id','inning','batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))

strikerate_inning['Strike Rate'] = strikerate_inning['batsman_runs']/strikerate_inning['ball']*100

strikerate_inning.reset_index(inplace = True)

strikerate_inning = strikerate_inning[strikerate_inning['inning'] <= 2]

strikerate_inning.reset_index(inplace = True)

plt.rcParams['figure.figsize'] = 20,15

sns.scatterplot(x = 'index',y = 'Strike Rate',hue = 'inning',data = strikerate_inning[strikerate_inning['batsman_runs'] > 20],hue_norm=(1, 2))

#strikerate_inning[strikerate_inning['batsman_runs'] > 20].plot(kind = 'scatter',x = 'index',y = 'Strike Rate')

plt.xlabel('Index',fontsize = 20)

plt.ylabel('Strike Rate',fontsize = 20)

plt.title('Strike rate in each Inning',fontsize = 20)

plt.show()

#Provide color using innings seaborn

strikerate_inning['inning'].unique()

player_strikerate_inning = pd.DataFrame(deliveries.groupby(['inning','batsman']).agg({'batsman_runs' : 'sum','ball' : 'count'}))

player_strikerate_inning['Strike Rate'] = player_strikerate_inning['batsman_runs']/player_strikerate_inning['ball']*100

player_strikerate_inning.reset_index(inplace = True)

player_strikerate_inning = player_strikerate_inning[player_strikerate_inning['inning'] <= 2]

player_strikerate_inning = player_strikerate_inning.sort_values(by = ['Strike Rate'],ascending = False)
# Taking only players who have scored more than 1500 runs in either innings

plt.rcParams['figure.figsize'] = 35,35

f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

player_strikerate_inning_one = player_strikerate_inning[player_strikerate_inning['inning'] == 1]

colors = cm.rainbow(np.linspace(0,1,len(player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500])))

player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500].plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax1)

ax1.set_ylabel('Strike Rate',fontsize = 20)

ax1.set_xlabel('')

ax1.set_title('Strike Rate 1st innings',fontsize = 20)



player_strikerate_inning_one = player_strikerate_inning[player_strikerate_inning['inning'] == 2]

colors = cm.rainbow(np.linspace(0,1,len(player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500])))

player_strikerate_inning_one[player_strikerate_inning_one['batsman_runs'] > 1500].plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax2)

ax2.set_ylabel('Strike Rate',fontsize = 20)

ax2.set_title('Strike Rate 2nd innings',fontsize = 20)

plt.xlabel('Batsman Name',fontsize = 20)

plt.show()
strikerate_result = pd.DataFrame(aggregatedata.groupby(['match_id','winner','batsman','batting_team']).agg({'batsman_runs' : 'sum','ball' : 'count'}))

strikerate_result.reset_index(inplace = True)

def win(x):

    if x['winner'] == x['batting_team'] :

        return 'Yes'

    else:

        return 'No'

strikerate_result['Win'] = strikerate_result.apply(win, axis=1)



strikerate_result['Strike Rate'] = strikerate_result['batsman_runs']/strikerate_result['ball']*100

strikerate_result.reset_index(inplace = True)

strikerate_result.reset_index(inplace = True)

plt.rcParams['figure.figsize'] = 20,15

sns.scatterplot(x = 'index',y = 'Strike Rate',hue = 'Win',data = strikerate_result[strikerate_result['batsman_runs'] > 20])

#strikerate_result[strikerate_result['batsman_runs'] > 20].plot(kind = 'scatter',x = 'index',y = 'Strike Rate')

plt.xlabel('Index',fontsize = 20)

plt.ylabel('Strike Rate',fontsize = 20)

plt.title('Strike rate and match result',fontsize = 20)

plt.show()

#Provide color using Win seaborn
# Taking only players who have scored more than 1500 runs in either innings

strikerate_result_win = pd.DataFrame(aggregatedata.groupby(['winner','batsman','batting_team']).agg({'batsman_runs' : 'sum','ball' : 'count'}))

strikerate_result_win.reset_index(inplace = True)

def win(x):

    if x['winner'] == x['batting_team'] :

        return 'Yes'

    else:

        return 'No'

strikerate_result_win['Win'] = strikerate_result_win.apply(win, axis=1)



strikerate_result_win.reset_index(inplace = True)

strikerate_result_win.head()

strikerate_result_win = pd.DataFrame(strikerate_result_win.groupby(['batsman','Win']).agg({'batsman_runs' : 'sum','ball' : 'sum'}))

strikerate_result_win['Strike Rate'] = strikerate_result_win['batsman_runs']/strikerate_result_win['ball']*100

strikerate_result_win.reset_index(inplace = True)

strikerate_result_win = strikerate_result_win.sort_values(by = ['Strike Rate'],ascending = False)

strikerate_result_win.head()

plt.rcParams['figure.figsize'] = 25,25



f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

strikerate_result_winn = strikerate_result_win[strikerate_result_win['Win'] == 'Yes']

colors = cm.rainbow(np.linspace(0,1,10))

strikerate_result_winn[strikerate_result_winn['batsman_runs'] > 1500].head(10).plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax1)

ax1.set_ylabel('Strike Rate',fontsize = 20)

ax1.set_xlabel('')

ax1.set_title('Strike Rate during Wins',fontsize = 20)



strikerate_result_lose = strikerate_result_win[strikerate_result_win['Win'] == 'No']

strikerate_result_lose[strikerate_result_lose['batsman_runs'] > 1500].head(10).plot(kind = 'bar',y = 'Strike Rate',x = 'batsman',color = colors,legend = '',fontsize = 10,ax = ax2)

ax2.set_ylabel('Strike Rate',fontsize = 20)

ax2.set_title('Strike Rate during Losses',fontsize = 20)

plt.xlabel('Batsman Name',fontsize = 20)

plt.show()

batsmen_average = pd.DataFrame(deliveries.groupby(['batsman']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))

batsmen_average['Average'] = batsmen_average['batsman_runs']/batsmen_average['player_dismissed']

batsmen_average = batsmen_average.sort_values(by = 'Average',ascending = False)

batsmen_average[batsmen_average['batsman_runs'] > 2500]
plt.rcParams['figure.figsize'] = 15,10

colors = cm.rainbow(np.linspace(0,1,len(batsmen_average[batsmen_average['batsman_runs'] > 2500])))

batsmen_average[batsmen_average['batsman_runs'] > 2500].plot(kind = 'bar',y = 'Average',

                                                                                color = colors,legend = '',fontsize = 10)

plt.xlabel('Batsman Name',fontsize = 20)

plt.ylabel('Average',fontsize = 20)

plt.show()
batsmen_averagesr = pd.merge(batsmen_strikerate_season,batsmen_average,left_on = 'batsman',right_on = 'batsman')

batsmen_averagesr.reset_index(inplace = True)

batsmen_averagesr['Category'] = batsmen_averagesr['batsman_runs_x'].apply(lambda x: 1 if x <= 250 

                                                                         else( 2 if x<=500 

                                                                         else( 3 if x<=1000  

                                                                         else( 4 if x<=1500

                                                                         else( 5 if x<=2000

                                                                         else( 6 if x<=2500 

                                                                         else 7))))))

batsmen_averagesr['Category'].unique()
fig, ax = plt.subplots()

categories = np.unique(batsmen_averagesr['Category'])

colors = np.linspace(0, 1, len(categories))

colordict = dict(zip(categories, colors))  



batsmen_averagesr["Color"] = batsmen_averagesr['Category'].apply(lambda x: colordict[x])

#ax.scatter(batsmen_averagesr['Strike Rate'],batsmen_averagesr['Average'],c =batsmen_averagesr['Color'])

sns.scatterplot(x = 'Strike Rate',y = 'Average',hue = 'Category',data = batsmen_averagesr)

plt.xlabel('Strike Rate')

plt.ylabel('Average')



plt.show()
batsmen_average_season = pd.DataFrame(aggregatedata.groupby(['season','batsman']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))

batsmen_average_season['Average'] = batsmen_average_season['batsman_runs']/batsmen_average_season['player_dismissed']

batsmen_average_season = batsmen_average_season.sort_values(by = ['season','Average'],ascending = False)

batsmen_average_season.reset_index(inplace = True)

batsmen_average_season[batsmen_average_season['batsman_runs'] > 300].head()
colors = cm.rainbow(np.linspace(0,1,10))

plt.rcParams['figure.figsize'] = 10,5

for title,group in batsmen_average_season.groupby('season'):

    group[group['batsman_runs'] > 300].head(10).plot(x = 'batsman',y = 'Average',kind = 'bar',legend = '',

                                                     color = colors,fontsize = 10)

    plt.xlabel('Batsman Name',fontsize = 20)

    plt.ylabel('Average',fontsize = 20)

    plt.title('Top 10 Average in Season %s '%title,fontsize = 20)

plt.show()
average_inning = pd.DataFrame(deliveries.groupby(['inning','batsman']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))

average_inning['Average'] = average_inning['batsman_runs']/average_inning['player_dismissed']

average_inning.reset_index(inplace = True)

average_inning = average_inning[average_inning['inning'] <= 2]

average_inning.reset_index(inplace = True)

plt.rcParams['figure.figsize'] = 15,5

sns.scatterplot(x = 'index',y = 'Average',hue = 'inning',data = average_inning[average_inning['batsman_runs'] > 500],hue_norm=(1, 2))

#average_inning[average_inning['batsman_runs'] > 500].plot(kind = 'scatter',x = 'index',y = 'Average')

#Taking more than 500 so that we make usre we have included only the batsmen

plt.xlabel('Index',fontsize = 20)

plt.ylabel('Average',fontsize = 20)

plt.title('Average in each Inning',fontsize = 20)



plt.show()

#Provide color using innings seaborn
average_inning = average_inning.sort_values(by = ['Average'],ascending = False)



plt.rcParams['figure.figsize'] = 35,35



f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

average_inning_one = average_inning[average_inning['inning'] == 1]

average_inning_one = average_inning_one[average_inning_one['player_dismissed'] != 0 ]

colors = cm.rainbow(np.linspace(0,1,15))

average_inning_one[average_inning_one['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax1)

ax1.set_ylabel('Average',fontsize = 20)

ax1.set_xlabel('')

ax1.set_title('Average 1st innings',fontsize = 20)



average_inning_two = average_inning[average_inning['inning'] == 2]

average_inning_two = average_inning_two[average_inning_two['player_dismissed'] != 0 ]

average_inning_two[average_inning_two['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax2)

ax2.set_ylabel('Average',fontsize = 20)

ax2.set_title('Average 2nd innings',fontsize = 20)

plt.xlabel('Batsman Name',fontsize = 20)

plt.show()
average_result = pd.DataFrame(aggregatedata.groupby(['winner','batsman','batting_team']).agg({'batsman_runs' : 'sum','player_dismissed' : 'count'}))

average_result.reset_index(inplace = True)

def win(x):

    if x['winner'] == x['batting_team'] :

        return 'Yes'

    else:

        return 'No'

average_result['Win'] = average_result.apply(win, axis=1)



average_result = pd.DataFrame(average_result.groupby(['batsman','Win']).agg({'batsman_runs' : 'sum', 'player_dismissed' : 'sum' }))



average_result['Average'] = average_result['batsman_runs']/average_result['player_dismissed']

average_result.reset_index(inplace = True)

average_result.reset_index(inplace = True)

average_result = average_result[average_result['player_dismissed'] != 0]

plt.rcParams['figure.figsize'] = 15,5



sns.scatterplot(x = 'index',y = 'Average',hue = 'Win',data = average_result[average_result['batsman_runs'] > 500],hue_norm=(1, 2))

#average_result[average_result['batsman_runs'] > 500].plot(kind = 'scatter',x = 'index',y = 'Average')

plt.xlabel('Index',fontsize = 20)

plt.ylabel('Average',fontsize = 20)

plt.title('Strike rate and match result',fontsize = 20)

plt.show()
average_result = average_result.sort_values(by = ['Average'],ascending = False)



plt.rcParams['figure.figsize'] = 35,35



f, (ax1, ax2) = plt.subplots(2, 1,sharey = True)

average_result_win = average_result[average_result['Win'] == 'Yes']

colors = cm.rainbow(np.linspace(0,1,15))

average_result_win[average_result_win['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax1)

ax1.set_ylabel('Average',fontsize = 20)

ax1.set_xlabel('')

ax1.set_title('Average during wins ',fontsize = 20)



average_result_lose = average_result[average_result['Win'] == 'No']

average_result_lose[average_result_lose['batsman_runs'] > 1000].head(15).plot(kind = 'bar',y = 'Average',x = 'batsman',color = colors,legend = '',fontsize = 12,ax = ax2)

ax2.set_ylabel('Average',fontsize = 20)

ax2.set_title('Average during losses',fontsize = 20)

plt.xlabel('Batsman Name',fontsize = 20)

plt.show()