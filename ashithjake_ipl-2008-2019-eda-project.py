# Import Libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Read Data

#Match by match data

matches = pd.read_csv('/kaggle/input/ipldata/matches.csv') 

#Ball by ball data

deliveries = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')
# Fixing some of the discrepant data present in deliveries dataframe

def fix_discrepant_data(df):

    if df[0] in range(7894,7954):

        df[15] = df[15] - df[16]

        df[17] = df[17] - df[16]

        return df

    elif df[0] in range(11137,11415):

        df[15] = df[15] - df[16]

        df[17] = df[17] - df[16]

        return df

    else:

        return df

deliveries = deliveries.apply(fix_discrepant_data,axis=1)
#Create new dataframe

most_runs = pd.DataFrame()



# From the 'deliveries' dataframe groupby rows based off 'batsman' column, perform sum on them, fetch only 

# the 'batsman_runs' column, sort them and fetch top 10 results

most_runs['Total Runs'] = deliveries.groupby('batsman').sum()['batsman_runs'].sort_values(ascending = False).head(10)



#Give a name to the index and reset the index to make it a column

most_runs.index.names = ['Batsman']

most_runs.reset_index(inplace=True)



#Plot the graph

plt.figure(figsize=(18,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Most Runs',fontdict=font)

ax = sns.barplot(x='Batsman',y='Total Runs',data = most_runs,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()-500), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

#Create new dataframe

individual_runs = pd.DataFrame()



# From the 'deliveries' dataframe groupby rows based off 'match id' and'batsman' columns, perform sum on them, fetch only 

# the 'batsman_runs' column, sort them and fetch top 10 results

individual_runs['Individual Score'] = deliveries.groupby(['match_id','batsman']).sum()['batsman_runs'].sort_values(ascending = False).head(10)



#Give names to the index and reset the index to make it a column

individual_runs.index.names = ['id','Batsman']

individual_runs.reset_index(inplace=True)



#Merge the 'matches' dataframe to get the 'season' info

individual_runs = pd.merge(individual_runs,matches,on='id')[['Batsman','season','Individual Score']]

individual_runs['Batsman - season'] = individual_runs['Batsman'] + ' - ' +individual_runs['season'].astype(str)



#Plot the graph

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Highest Individual Score',fontdict=font)

ax = sns.barplot(x='Batsman - season',y='Individual Score',data = individual_runs,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()-15), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

#Create new dataframes

runs = pd.DataFrame()

dissmissed = pd.DataFrame()

avg = pd.DataFrame()



#Get the total runs scored by each batsman

runs['total_runs'] = deliveries.groupby('batsman').sum()['batsman_runs']



#Get the number of times each batsman have been dismissed

dissmissed['total_dismissed'] = deliveries['player_dismissed'].value_counts()



#Join the 2 dataframes

runs = runs.join(dissmissed)

runs = runs.dropna()



#Calculate Average,sort them and fetch the top 10 averages

runs['avg'] = runs['total_runs']/runs['total_dismissed']

avg['Average'] = runs['avg'].sort_values(ascending=False).head(10)



#Give name to the index and reset the index to make it a column

avg.index.names = ['Batsman']

avg.reset_index(inplace=True)



#Plot the graph

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Best Batting Average',fontdict=font)

ax = sns.barplot(x='Batsman',y='Average',data = avg,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x() + p.get_width() / 2., p.get_height()-5), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

#Create new dataframes

runs = pd.DataFrame()

balls = pd.DataFrame()

str_rate = pd.DataFrame()



#Collect total runs and balls faced for each batsman

runs['total_runs'] = deliveries.groupby('batsman').sum()['batsman_runs']

balls['ball_Faced'] = deliveries['batsman'].value_counts()



#Join the 2 dataframes

runs = runs.join(balls)



#Calculate Strike Rate,sort them and fetch the top 10 Strike Rates

runs['Strike Rate'] = runs.apply(lambda x: 0 if x[1]<150 else (x[0]/x[1])*100 ,axis=1)

str_rate['Strike Rate'] = runs['Strike Rate'].sort_values(ascending=False).head(10)



#Give name to the index and reset the index to make it a column

str_rate.index.names = ['Batsman']

str_rate.reset_index(inplace=True)





#Plot the graph

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Best Batting Strike Rate',fontdict=font)

ax = sns.barplot(x='Batsman',y='Strike Rate',data = str_rate,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x() + p.get_width() / 2., p.get_height()-15), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

#Create new dataframe

wickets = pd.DataFrame()

dismissal = pd.DataFrame()



#Collect the dismissals by each bowler and group them to get the count of wickets

dismissal = deliveries[(deliveries['player_dismissed'].notnull()) & 

               (~deliveries['dismissal_kind'].isin(['run out','retired hurt','obstructing the field']))]

wickets['Wickets'] = dismissal.groupby('bowler').count()['player_dismissed'].sort_values(ascending=False).head(10)



#Give name to the index and reset the index to make it a column

wickets.index.names = ['Bowler']

wickets.reset_index(inplace=True)





#Plot the graph

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Most Wickets',fontdict=font)

ax = sns.barplot(x='Bowler',y='Wickets',data = wickets,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()-15), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

#Create new dataframe

runs = pd.DataFrame()

dots = pd.DataFrame()



#Collect the count of dot balls

runs = deliveries[deliveries['total_runs'] == 0]

dots['Dots'] = runs.groupby('bowler').count()['total_runs'].sort_values(ascending=False).head(10)



#Give name to the index and reset the index to make it a column

dots.index.names = ['Bowler']

dots.reset_index(inplace=True)





#Plot the graph

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Most Dot Balls',fontdict=font)

ax = sns.barplot(x='Bowler',y='Dots',data = dots,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()-150), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

#Create new dataframe

runs_conceded = pd.DataFrame()

total_wickets = pd.DataFrame()

dismissal = pd.DataFrame()

bowl_avg = pd.DataFrame()



#Calculate total runs conceded by each bowlers

runs_conceded = deliveries.groupby('bowler').agg(['sum','count'])['total_runs']

runs_conceded = runs_conceded[runs_conceded['count']>300]

runs_conceded.rename(columns={"sum": "Total Runs Conceded"},inplace = True)



#Calculate total wickets taken by each bowlers

dismissal = deliveries[(deliveries['player_dismissed'].notnull()) & 

               (~deliveries['dismissal_kind'].isin(['run out','retired hurt','obstructing the field']))]

total_wickets['Wickets'] = dismissal.groupby('bowler').count()['player_dismissed']



#join the two dataframes

runs_conceded = runs_conceded.join(total_wickets)



#Calculate the bowling average

runs_conceded['Bowling Average'] = runs_conceded['Total Runs Conceded']/runs_conceded['Wickets']

bowl_avg['Bowling Average'] = runs_conceded['Bowling Average'].sort_values().head(10)





#Give name to the index and reset the index to make it a column

bowl_avg.index.names = ['Bowler']

bowl_avg.reset_index(inplace=True)





#Plot the graph

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Best Bowling Average',fontdict=font)

ax = sns.barplot(x='Bowler',y='Bowling Average',data = bowl_avg,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x() + p.get_width() / 2., p.get_height()-2), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

#Considering only the current teams

current_teams = ['Mumbai Indians','Kolkata Knight Riders','Chennai Super Kings','Kings XI Punjab',

                 'Royal Challengers Bangalore','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad']



#Create new dataframe

toss_winners = pd.DataFrame()



#Get the number of tosses won

toss_winners['Toss Won'] = matches['toss_winner'].value_counts()

toss_winners.index.names = ['Teams']

toss_winners.reset_index(inplace=True)



toss_winners = toss_winners[toss_winners['Teams'].isin(current_teams)]



# Plot

plt.figure(figsize=(16,10))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

cmap = plt.get_cmap("rainbow")

colors = cmap(np.array([75,100, 125,150, 175, 200,225, 250]))

textprops = {"fontsize":15,"color":"indigo"}

plt.title('Toss Winners',fontdict=font)

plt.pie(toss_winners['Toss Won'], labels=toss_winners['Teams'], autopct='%1.1f%%', startangle=140,textprops=textprops, colors=colors)



plt.axis('equal')

plt.show()

#Considering only the current teams



current_teams = ['Mumbai Indians','Kolkata Knight Riders','Chennai Super Kings','Kings XI Punjab',

                 'Royal Challengers Bangalore','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad']



#Create new dataframe

toss_winners = pd.DataFrame()



#Get the number of tosses won

toss_winners['Matches Won'] = matches['winner'].value_counts()

toss_winners.index.names = ['Teams']

toss_winners.reset_index(inplace=True)



toss_winners = toss_winners[toss_winners['Teams'].isin(current_teams)]



# Plot

plt.figure(figsize=(16,10))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

cmap = plt.get_cmap("rainbow")

colors = cmap(np.array([75,100, 125,150, 175, 200,225, 250]))

textprops = {"fontsize":15,"color":"indigo"}

plt.title('Maximum Match Winners',fontdict=font)

plt.pie(toss_winners['Matches Won'], labels=toss_winners['Teams'], autopct='%1.1f%%', startangle=140,textprops=textprops,colors = colors)



plt.axis('equal')

plt.show()
#Considering only the current teams

current_teams = ['Mumbai Indians','Kolkata Knight Riders','Chennai Super Kings','Kings XI Punjab',

                 'Royal Challengers Bangalore','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad']



#Create new dataframe

score_dist = pd.DataFrame()





score_dist['Total Score'] = deliveries.groupby(['match_id','batting_team']).sum()['total_runs']

score_dist.index.names = ['Match ID','Teams']

score_dist.reset_index(inplace=True)

score_dist = score_dist[score_dist['Teams'].isin(current_teams)]



plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

meanprops ={"marker":"o",

            "markerfacecolor":"indigo", 

            "markeredgecolor":"black",

            "markersize":"10"

           }

plt.title('Score Distribution',fontdict=font)

ax = sns.boxplot(x='Teams',y='Total Score',data=score_dist,showmeans=True,meanprops=meanprops)

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)

plt.show()
#Considering only the current teams

current_teams = ['Mumbai Indians','Kolkata Knight Riders','Chennai Super Kings','Kings XI Punjab',

                 'Royal Challengers Bangalore','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad']



#function to get the info whether the winning team batted first or not

def winner_bat_first_or_not(df):

    if df[0] == df[2]:

        return df[1]

    else:

        if df[1] == 'field':

            return 'bat'

        else:

            return 'field'



#Create new dataframe

winner = pd.DataFrame()



winner = matches[['toss_winner','toss_decision','winner']].copy()

winner['winner_bat_first_or_field'] = winner.apply(winner_bat_first_or_not,axis=1)

winner = winner[winner['winner'].isin(current_teams)]



#Plot

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }



plt.title('Win Distribution',fontdict=font)

ax = sns.countplot(x='winner',data=winner,hue='winner_bat_first_or_field')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x() + p.get_width() / 2., p.get_height()-1), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

plt.show()
#Considering only the current teams

current_teams = ['Mumbai Indians','Kolkata Knight Riders','Chennai Super Kings','Kings XI Punjab',

                 'Royal Challengers Bangalore','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad']





#Create new dataframe

toss_winner = pd.DataFrame()



toss_winner = matches[['toss_winner','toss_decision']]

toss_winner = toss_winner[toss_winner['toss_winner'].isin(current_teams)]



#Plot

plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }



plt.title('Toss Distribution',fontdict=font)

ax = sns.countplot(x='toss_winner',data=toss_winner,hue='toss_decision')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x() + p.get_width() / 2., p.get_height()-1), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)

plt.show()
#Create new dataframe

toss_decision = pd.DataFrame()



#Collect season wise counts of toss decision

toss_decision['toss_decision_count'] = matches.groupby('season')['toss_decision'].value_counts()

toss_decision.index.names = ['Season','Toss Decision']

toss_decision.reset_index(inplace=True)

toss_decision = toss_decision.pivot_table(values='toss_decision_count',columns=['Toss Decision'],index=['Season'])



stacked_data = toss_decision.apply(lambda x: x*100/sum(x), axis=1)





#Plot

plt.figure()

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }



ax = stacked_data.plot(kind="bar", stacked=True)

ax.set_title('Toss Distribution Over Seasons',fontdict=font)

plt.ylabel('Percentage')

ax.legend(loc=(1.1,0.8))

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=14)
