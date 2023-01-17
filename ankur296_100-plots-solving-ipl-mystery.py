import numpy as np

import seaborn as sns

sns.set_style('darkgrid')

sns.set(color_codes=True)

from datetime import datetime

import pandas as pd

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import re

from string import Template

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/ipldata/matches.csv',parse_dates=['date'], dayfirst=True)

print('='*20 + 'Exploratory Data Analysis' + '='*20)

print('='*20 + 'SHAPE' + '='*20)

print(df.shape)

print('\n\n')

print('='*20 + 'INFO' + '='*20)

df.info()

print('\n\n')

print('='*20 + 'DESCRIBE' + '='*20)

print(df.describe())

print('\n\n')



print('='*20 + 'NA CHECK' + '='*20)

print(df.isna().any())

print(df.isna().sum())

print('\n\n')

print('='*20 + 'UNIQUE VALUES' + '='*20)

# I will only check those col for unique values where I'm not sure what possible values can the col take. For others it would be a sanity check, like wickets can't be more than 10 etc

print('\n Result')

print(df.result.unique())

print('\n toss_decision')

print(df.toss_decision.unique())

print('\n winner')

print(df.winner.unique())

print('\n win_by_runs')

print(df.win_by_runs.unique())

print('\n win_by_wickets')

print(df.win_by_wickets.unique())
# Drop ump3 col 

df1 = df.drop(columns='umpire3')

# Sort by season, date (2nd level sort),city

df1 = df1.sort_values(by=['season','date','city'])



df1.replace({'Rising Pune Supergiant':'Rising Pune Supergiants', 'Delhi Daredevils': 'Delhi Capitals', 'Bengaluru':'Bangalore'}, inplace=True)



# Removing leading/trailing whitespaces and changing to upper case

# TODO: There has to be a better way to do this!

df1.city = df1.city.str.strip().str.upper()

df1.team1 = df1.team1.str.strip().str.upper()

df1.team2 = df1.team2.str.strip().str.upper()

df1.toss_winner = df1.toss_winner.str.strip().str.upper()

df1.toss_decision = df1.toss_decision.str.strip().str.upper()

df1.result = df1.result.str.strip().str.upper()

df1.winner = df1.winner.str.strip().str.upper()

df1.player_of_match = df1.player_of_match.str.strip().str.upper()

df1.venue = df1.venue.str.strip().str.upper()

df1.umpire1 = df1.umpire1.str.strip().str.upper()

df1.umpire2 = df1.umpire2.str.strip().str.upper()





# df1['team1'].value_counts()

df1.head()

# Let's address all team names by their initials. This will result in cleaner plot labels/ticks

df1.replace({

'CHENNAI SUPER KINGS': 'CSK',

'DELHI CAPITALS': 'DC',

'KINGS XI PUNJAB': 'KXIP',

'KOLKATA KNIGHT RIDERS': 'KKR',

'MUMBAI INDIANS': 'MI',

'RAJASTHAN ROYALS': 'RR',

'ROYAL CHALLENGERS BANGALORE': 'RCB',

'SUNRISERS HYDERABAD': 'SRH',

'DECCAN CHARGERS': 'DCH',

'GUJARAT LIONS':'GL',

'KOCHI TUSKERS KERALA': 'KTK',

'RISING PUNE SUPERGIANTS':'RPS',

'PUNE WARRIORS':'PW'

}, inplace=True)

#We would need a team list at some point

team1_elements = df1.team1.value_counts().index.tolist()

team2_elements = df1.team2.value_counts().index.tolist()

team_list = list(set(team1_elements + team2_elements))

team_list.sort()

print(team_list)

print("Total number of teams: "+ str(len(team_list)))
trivial_teams = [

 'DCH',

 'GL',

 'KTK',

 'RPS',

 'PW']



# trivial_teams
imp_teams = [team for team in team_list if team not in trivial_teams]



# declare vars holding these team names

csk = 'CSK'

dc = 'DC'

kxip = 'KXIP'

kkr ='KKR'

mi = 'MI'

rr = 'RR'

rcb = 'RCB'

srh = 'SRH'

dch = 'DCH'

gl = 'GL'

ktk = 'KTK'

rps = 'RPS'

pw = 'PW'



imp_teams
champ_teams = [csk,kkr,mi]
# line/marker colors matching the uniforms of the imp teams

imp_cities = ['MUMBAI',

'BANGALORE',  

'KOLKATA',

'DELHI',

'CHENNAI']

imp_cities.sort()

# ipl_city_palette = ["#FFFF3C", "#00008B", "#2E0854","#004BA0","#EC1C24"]

ipl_line_palette = ["#FFFF3C", "#00008B",  "#2E0854","#ED1B24","#004BA0", "#EC1C24","pink","#FF822A"]

ipl_marker_palette = ["#0081E9", "#EF1B23",  "#B3A123","#DCDDDF","#D1AB3E", "#2B2A29","#254AA5","#000000"]

ipl_city_palette = {'BANGALORE':"#EC1C24", 'CHENNAI': "#FFFF3C", 'DELHI':"#00008B" , 'KOLKATA':"#2E0854" , 'MUMBAI':"#004BA0"}

ipl_team_palette = {csk:"#FFFF3C", dc:"#00008B", kxip:"#ED1B24", kkr:"#2E0854",mi:"#004BA0", rr:"pink",rcb:"#EC1C24",srh:"#FF822A"}



imp_cities


def plot(xlabel, ylabel, title,  y_lim, y_init = 0, y_interval=10):

    """custom func to prevent repetition while plotting matplotlib plots"""

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    plt.yticks(range(y_init, y_lim + y_interval, y_interval))

    plt.show()



def plot_sns(g, xlabel="", ylabel="", title="", y = 1.07 , y_lim=100, y_init = 0, y_interval=10, rot = 90,face_grid=True):

    """custom func to prevent repetition while plotting Seaborn plots"""

    if face_grid is True:

        g.fig.suptitle(title, y = y)

    else:

        g.set(title = title)

    g.set(xlabel = xlabel, ylabel = ylabel)

    plt.yticks(range(y_init, y_lim + y_interval, y_interval))

    plt.xticks(rotation=rot)

    plt.show()
dict_city_homeground = {

    'CHANDIGARH': kxip, 

    'MOHALI': kxip,

    'DHARAMSALA': kxip,

    'CHENNAI': csk, 

    'BANGALORE': rcb, 

    'HYDERABAD': srh,

    'VISAKHAPATNAM': dch,

    'RAJKOT': gl, 

    'PUNE': rps, 

    'KOCHI': ktk, 

    'MUMBAI': mi, 

    'KOLKATA': kkr, 

    'JAIPUR': rr,

    'DELHI': dc

}



# create a new column indicating which is the home team for the match

df1['home_team'] = None

        

for row in df1.itertuples():

     df1.at[row.Index, 'home_team'] = dict_city_homeground.get(df1.at[row.Index,'city'], "")

        

df1.tail()

team1vc = df1[~df1.team1.isin(trivial_teams)].team1.value_counts()

team2vc = df1[~df1.team2.isin(trivial_teams)].team2.value_counts()

matches_per_team = (team1vc + team2vc).sort_index()
df1['toss_and_match_winners_same'] = df1.winner == df1.toss_winner

df1['home_team_victorious'] = df1.winner == df1.home_team

def get_match_loser(row):

    loser = ''

    if row['winner'] == row['team1']:

        loser = row['team2']

    else:

        loser = row['team1']

    return loser



def get_toss_loser(row):

    loser = ''

    if row['toss_winner'] == row['team1']:

        loser = row['team2']

    else:

        loser = row['team1']

    return loser



df1['loser'] = df1.apply(get_match_loser, axis = 1)

df1['toss_loser'] = df1.apply(get_toss_loser, axis = 1)



match_winners = df1.winner.value_counts().drop(labels=trivial_teams).sort_index()

match_losers = df1.loser.value_counts().drop(labels=trivial_teams).sort_index()

toss_winners = df1.toss_winner.value_counts().drop(labels=trivial_teams).sort_index()

toss_losers = df1.toss_loser.value_counts().drop(labels=trivial_teams).sort_index()

df1.head()
df1['day'] = pd.DatetimeIndex(df1['date']).day_name()

day_vc = df1['day'].value_counts()



order=['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']

g = sns.catplot(x="day", data = df1, kind="count", order = order, height=5, aspect = 1.5)

plot_sns(g, ylabel = "No of matches",title ="Matches played on each day of the week", y_lim = 160, y_interval = 20)

def calc_match_ratio_of_day(team):

    days = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']

    match_ratio_per_day = []

    is_team_involved = (df1.team1==team) | (df1.team2==team)

    for day in days:

        match_ratio_per_day.append( len( df1[ is_team_involved & (df1.day==day) ]) *100 // len(df1[ is_team_involved ]))

    

    return match_ratio_per_day



x_days = ['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']





plt.figure(figsize=(12,5))

plt.plot(x_days, calc_match_ratio_of_day(mi), marker='o', markerfacecolor='b', markersize=7, color='skyblue', linewidth=4,label='MI')

plt.plot(x_days, calc_match_ratio_of_day(csk),marker=(8,2,0), markerfacecolor='orange', markersize=9, color='y', linewidth=3, linestyle="--",label='CSK')

plt.plot(x_days, calc_match_ratio_of_day(kkr), marker='d', markerfacecolor='purple', markersize=5, color='darkblue', linewidth=2,linestyle="-.",label='KKR')

plt.plot(x_days, calc_match_ratio_of_day(rcb), marker='x', markerfacecolor='m', markersize=5, color='r', linewidth=1,linestyle=":",label='RCB')

plt.ylim([0,30])

plt.legend()

plt.ylabel('%age of matches played')

plt.title('Matches played on different days')

plt.show()

def calc_win_ratio_of_day(team):

    days = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']

    win_ratio_per_day = []

    for day in days:

        win_ratio_per_day.append( len( df1[ (df1.winner==team) & (df1.day==day) ]) *100 // len(df1[df1.winner==team]))

    

    return win_ratio_per_day



x_days = ['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']



plt.figure(figsize=(12,5))

plt.plot(x_days, calc_win_ratio_of_day(mi), marker='o', markerfacecolor='b', markersize=7, color='skyblue', linewidth=4,label='MI')

plt.plot(x_days, calc_win_ratio_of_day(csk),marker=(8,2,0), markerfacecolor='orange', markersize=9, color='#FFDB58', linewidth=3, linestyle="--",label='CSK')

plt.plot(x_days, calc_win_ratio_of_day(kkr), marker='d', markerfacecolor='purple', markersize=5, color='darkblue', linewidth=2,linestyle="-.",label='KKR')

plt.plot(x_days, calc_win_ratio_of_day(rcb), marker='x', markerfacecolor='m', markersize=5, color='r', linewidth=1,linestyle=":",label='RCB')



plt.ylim([0,30])

plt.legend()

plt.ylabel('%age of matches won')

plt.title('Performance on different days')

plt.show()

fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20,10), sharex=False, sharey=True)



def add_subplot(ax=ax, x=x_days , team='Mumbai Indians', m='o', mc='b', ms=12, c='skyblue', lw=6, ls ="solid", label='', labelc='b'):

    ax.plot(x, calc_win_ratio_of_day(team), marker=m, markerfacecolor=mc, markersize=ms, color=c, linewidth=lw, linestyle=ls, label=label)

#     ax.set_yticks(range(0, 45, 5))

    ax.set_title(label,color=labelc)



add_subplot(ax=ax[0,0], team=mi,m='o',mc='b', c='skyblue',label='MI',labelc = 'b')

add_subplot(ax=ax[0,1], team=csk,m='X',mc='#E56717', c='#FFDB58',label='CSK',labelc = '#E56717')

add_subplot(ax=ax[0,2], team=kkr,m='D',mc='#FFDB58', c='darkblue', label='KKR',labelc = 'darkblue')

add_subplot(ax=ax[0,3], team=rcb,m='8',mc='black', ms=10, c='r', label='RCB',labelc = 'r')

add_subplot(ax=ax[1,0], team=rr,m='d',mc='b', c='pink',label='RR',labelc = 'b')

add_subplot(ax=ax[1,1], team=kxip,m='d',mc='#BCC6CC', c='r', label='KXIP',labelc = 'r')

add_subplot(ax=ax[1,2], team=srh,m='d',mc='black', c='#E56717', label='SRH',labelc = '#E56717')

add_subplot(ax=ax[1,3], team=dc,m='d',mc='r', c='Blue',label='DC',labelc = 'b')



# fig.suptitle('Performance on different days', fontsize=20)

plt.setp(ax[:, 0], ylabel='% of Matches Won')

fig.tight_layout()
plt.figure(figsize=(8,5))

g = sns.barplot(x=matches_per_team.index, y=matches_per_team, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g,ylabel = 'No of matches', title = 'Matches played by each team', 

     y_lim = matches_per_team.max(), y_interval = 20, face_grid=False)





plt.figure(figsize=(8,5))

g = sns.barplot(x=match_winners.index, y=match_winners, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g,ylabel = 'No of matches', title = 'Matches won by each team', 

     y_lim = match_winners.max(), y_interval = 20, face_grid=False)
match_win_ratio = (match_winners.sort_index()*100 // matches_per_team.sort_index())

plt.figure(figsize=(8,5))

g = sns.barplot(x=match_win_ratio.index, y = match_win_ratio,

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age of matches won",title ="Win %age of each team", y_lim = match_win_ratio.max(), 

         y_interval=10,  face_grid=False)



toss_win_ratio = (toss_winners*100 // matches_per_team.sort_index())

plt.figure(figsize=(8,5))

g = sns.barplot(x=toss_win_ratio.index, y = toss_win_ratio, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age of tosses won",title ="Toss %age of each team", y_lim = toss_win_ratio.max(), 

         y_interval=10,  face_grid=False)
match_win_ratio = (match_winners.sort_index()*100 // matches_per_team.sort_index()).sort_values(ascending=False)

toss_win_ratio = (toss_winners.sort_index()*100 // matches_per_team.sort_index()).sort_values(ascending=False)

match_win_ratio_col_name = "Matches Won %age"

toss_win_ratio_col_name = "Tosses Won %age"

df_combo = pd.DataFrame({match_win_ratio_col_name:match_win_ratio, toss_win_ratio_col_name:toss_win_ratio}).sort_values([match_win_ratio_col_name], ascending = False)

# we shud pass figsize as an argument instead of calling it separately as we are not operating on a single column, instead multiple columns

df_combo.plot(color=["SkyBlue","IndianRed"],kind='bar', grid=True, figsize=(20,7))

plot(xlabel = 'Teams', ylabel = '%age of matches/tosses won', title = 'Matches Won %age vs Tosses Won %age', 

     y_lim = df_combo[[match_win_ratio_col_name,toss_win_ratio_col_name]].values.max())
same_toss_and_match_winners = df1['toss_winner'] == df1['winner']

match_winner_toss_winner = df1[same_toss_and_match_winners]['winner'].value_counts().sort_index()

match_winner_toss_winner_ratio = (match_winner_toss_winner *100 // toss_winners)

match_winner_toss_winner_ratio = match_winner_toss_winner_ratio.drop(labels=trivial_teams)

plt.figure(figsize=(8,5))

g = sns.barplot(x=match_winner_toss_winner_ratio.index, y = match_winner_toss_winner_ratio, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age",title ="%age of matches won after winning toss", y_lim = 75, 

         y_interval=10,  face_grid=False)

diff_toss_and_match_winners = df1['toss_winner'] != df1['winner']

match_loser_toss_winner = df1[diff_toss_and_match_winners]['toss_winner'].value_counts().sort_index()



match_loser_toss_winner_ratio = (match_loser_toss_winner *100 // toss_winners)

match_loser_toss_winner_ratio = match_loser_toss_winner_ratio.drop(labels=trivial_teams)

plt.figure(figsize=(8,5))



g = sns.barplot(x=match_loser_toss_winner_ratio.index, y = match_loser_toss_winner_ratio, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age",title ="%age of matches lost after winning toss", y_lim = 70, 

         y_interval=10,  face_grid=False)
match_winner_toss_loser = df1['toss_loser'] == df1['winner']

match_winner_toss_loser = df1[match_winner_toss_loser]['winner'].value_counts().sort_index()

match_winner_toss_loser_ratio = (match_winner_toss_loser*100 // toss_losers)

match_winner_toss_loser_ratio = match_winner_toss_loser_ratio.drop(labels=trivial_teams)



plt.figure(figsize=(8,5))



g = sns.barplot(x=match_winner_toss_loser_ratio.index, y = match_winner_toss_loser_ratio, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age",title ="%age of matches won after losing toss", y_lim = 70, 

         y_interval=10,  face_grid=False)
match_loser_toss_loser = df1['toss_loser'] == df1['loser']

match_loser_toss_loser = df1[match_loser_toss_loser]['loser'].value_counts().sort_index()



match_loser_toss_loser_ratio = (match_loser_toss_loser*100 // toss_losers)

match_loser_toss_loser_ratio = match_loser_toss_loser_ratio.drop(labels=trivial_teams)

plt.figure(figsize=(8,5))



g = sns.barplot(x=match_loser_toss_loser_ratio.index, y = match_loser_toss_loser_ratio, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age",title ="%age of matches lost after losing toss", y_lim = 70, 

         y_interval=10,  face_grid=False)

won_at_home_vc = df1[df1['winner'] == df1['home_team']].winner.value_counts()

played_at_home_vc = df1['home_team'].value_counts()



won_at_home_col = "Won at home"

played_at_home_col = "Played at home"

played_col = "Total Played"

df_win_home_ratio = pd.DataFrame({played_col: matches_per_team, played_at_home_col: played_at_home_vc, 

                                  won_at_home_col:won_at_home_vc}).sort_values([played_col], ascending=False)



df_win_home_ratio[won_at_home_col] = df_win_home_ratio[won_at_home_col].fillna(0).astype(np.int64)

df_win_home_ratio[played_at_home_col] = df_win_home_ratio[played_at_home_col].fillna(0).astype(np.int64)

df_win_home_ratio[played_col] = df_win_home_ratio[played_col].fillna(0).astype(np.int64)



df_win_home_ratio = df_win_home_ratio[0:8].sort_index()

# we shud pass figsize as an argument instead of calling it separately as we are not operating on a single column, instead multiple columns

df_win_home_ratio.plot(color=["SkyBlue","IndianRed",'green'],kind='bar', grid=True, figsize=(10,5))

plt.legend(loc="upper left", bbox_to_anchor=(1,1))

plot(xlabel='', ylabel = 'No of matches', title = 'Home ground analysis', 

     y_lim = df_win_home_ratio[played_col].values.max(), y_interval = 20)
home_matches_won_ratio = (df_win_home_ratio[won_at_home_col].sort_index()*100 // df_win_home_ratio[played_at_home_col].sort_index())

plt.figure(figsize=(8,5))

g = sns.barplot(x=home_matches_won_ratio.index, y = home_matches_won_ratio, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age",title ="%age of matches won on home ground", y_lim = 80, 

         y_interval=10,  face_grid=False)

# Plot no of matches per season

plt.figure(figsize=(10,5))

matches_per_season = df1.season.value_counts().sort_index()

# print(matches_per_season)

g = sns.barplot(x=matches_per_season.index, y = matches_per_season)

plot_sns(g, ylabel = "No of matches",title ="Matches in every season", y_lim = matches_per_season.max(), y_interval=10,  face_grid=False)



# Plot no of matches per season in which team decided to chase after winning toss

plt.figure(figsize=(10,5))

chases_per_season = df1[df1.toss_decision == "FIELD"].groupby('season')['toss_decision'].count()

# print(chases_per_season)

g = sns.barplot(x=chases_per_season.index, y = chases_per_season)

plot_sns(g, ylabel = "No of matches",title ="Decided to chase on winning toss", y_lim = chases_per_season.max(), y_interval=10,  face_grid=False)





# Plot which cities hosted most matches per season

plt.figure(figsize=(18,8))

matches_per_imp_cities = df1[df1.city.isin(imp_cities)]

y_lim = matches_per_imp_cities.groupby(['season','city']).count().max(level=0).id.max()

g = sns.countplot(x = matches_per_imp_cities.season, data = matches_per_imp_cities, hue = matches_per_imp_cities.city, palette = ipl_city_palette)

plot_sns(g, ylabel = "No of matches",title = "Cities hosting max matches", y_lim = y_lim, y_interval=4,  face_grid=False)





plt.figure(figsize=(18,5))

matches_per_city = df1.city.value_counts()

# print(matches_per_city)

g = sns.barplot(x=matches_per_city[matches_per_city >= 10].index, y = matches_per_city[matches_per_city >= 10])

plot_sns(g, ylabel = "No of matches",title ="Matches in every city", y_lim = matches_per_city[matches_per_city >= 10].max(), y_interval=10, face_grid=False)

plt.figure(figsize=(18,4))

matches_per_venue = df1.venue.value_counts()

g = sns.barplot(x=matches_per_venue[matches_per_venue >= 15].index, y = matches_per_venue[matches_per_venue >= 15])

plot_sns(g, ylabel = "No of matches",title ="Matches in every venue", y_lim = matches_per_venue[matches_per_venue >= 10].max(), y_interval=10, face_grid=False)

# Plot which teams won most matches per season

plt.figure(figsize=(18,8))

winners_per_season = df1[df1.winner.isin(champ_teams)].sort_values(by=['winner'])

y_lim = winners_per_season.groupby(['season','winner']).count().max(level=0).id.max()

g = sns.countplot(x = winners_per_season.season, data = winners_per_season, hue = winners_per_season.winner, palette = ipl_team_palette)

plot_sns(g, ylabel = "No of matches",title = "Teams winning max matches", y_lim = y_lim, y_interval=3,  face_grid=False)



# Plot who won most MOMs per season

plt.figure(figsize=(18,7))



s = df1.groupby(['season']).player_of_match.value_counts()

s = s.reset_index(name='no')

s = s[s.groupby(['season'])['no'].transform('max') == s['no']]

g = sns.barplot(x=s.player_of_match, y = s.no, hue=s.season, dodge=False)

plot_sns(g, ylabel = "No of MOM awards",title ="Players who won most MOMs/season", y_lim = s.no.max(), y_interval=1, face_grid=False)



plt.figure(figsize=(20,8))

mom_series = df1.player_of_match.value_counts()

# print(mom_series.head(30))

g = sns.barplot(x=mom_series[mom_series >= 10].index, y = mom_series[mom_series >= 10])

plot_sns(g, ylabel = "No of MOM awards",title ="Players who won MOM most times", y_lim = mom_series.max(), y_interval=5, face_grid=False)



season_list = df1.season.value_counts().sort_index().index.tolist()

season_winner_list = [

    rr,

    dch,

    csk,

    csk,

    kkr,

    mi,

    kkr,

    mi,

    srh,

    mi,

    csk,

    mi

]



season_runnerup_list = [

    csk,

    rcb,

    mi,

    rcb,

    csk,

    csk,

    kxip,

    csk,

    rcb,

    rps,

    srh,

    csk

]



df_summary = pd.DataFrame({'season': season_list, 'Winner': season_winner_list, 'Runner Up': season_runnerup_list})



fig, ax = plt.subplots(1,2, figsize=(15,5))



winner_vc = df_summary['Winner'].value_counts()

winner_vc.sort_values(ascending=False).plot(ax = ax[0], color="SkyBlue", kind='bar', grid=True)

ax[0].set_ylabel('Title Wins')

ax[0].set_yticks(range(0, 5, 1))



finalist_vc = df_summary.Winner.append(df_summary['Runner Up']).value_counts()

finalist_vc.sort_values(ascending=False).plot(ax = ax[1], color="SkyBlue", kind='bar', grid=True)

ax[1].set_ylabel('Finals Played')

ax[1].set_yticks(range(0, 10, 1))

    

fig.tight_layout()

csk_involved = (df1['team1'] == csk) | (df1['team2'] == csk)

mi_involved = (df1['team1'] == mi) | (df1['team2'] == mi)

kkr_involved = (df1['team1'] == kkr) | (df1['team2'] == kkr)

rcb_involved = (df1['team1'] == rcb) | (df1['team2'] == rcb)



mi_vs_csk_vc = df1[csk_involved & mi_involved]['winner'].value_counts()

mi_vs_kkr_vc = df1[kkr_involved & mi_involved]['winner'].value_counts()

mi_vs_rcb_vc = df1[rcb_involved & mi_involved]['winner'].value_counts()

csk_vs_kkr_vc = df1[csk_involved & kkr_involved]['winner'].value_counts()

csk_vs_rcb_vc = df1[csk_involved & rcb_involved]['winner'].value_counts()

kkr_vs_rcb_vc = df1[rcb_involved & kkr_involved]['winner'].value_counts()



fig, ax = plt.subplots(nrows = 1, ncols = 6, figsize=(20,8), sharey=True)



ax[0].set_ylabel('Matches Won')

ax[0].set_yticks(range(0, 30, 1))



def new_subplot(df = None, width = 0.3, title = "", ax = None, color = "SkyBlue", kind = 'bar', grid = True):

    df.plot(title = title, width = width, ax = ax, color = color, kind = kind, grid = grid)



new_subplot(mi_vs_csk_vc, title="MI vs CSK", ax = ax[0])

new_subplot(mi_vs_kkr_vc, title="MI vs KKR", ax = ax[1])

new_subplot(mi_vs_rcb_vc, title="MI vs RCB", ax = ax[2])

new_subplot(csk_vs_kkr_vc, title="CSK vs KKR", ax = ax[3])

new_subplot(csk_vs_rcb_vc, title="CSK vs RCB", ax = ax[4])

new_subplot(kkr_vs_rcb_vc, title="KKR vs RCB", ax = ax[5])



fig.tight_layout()



mi_vs_csk_mom_vc = df1[csk_involved & mi_involved]['player_of_match'].value_counts()

plt.figure(figsize=(15,5))

g = sns.barplot(x=mi_vs_csk_mom_vc.index, y = mi_vs_csk_mom_vc)

plot_sns(g, ylabel = "MOM awards",title ="MOM winners in MI vs CSK matches", y_lim = mi_vs_csk_mom_vc.max(), y_interval=1, face_grid=False)

mi_vs_kkr_mom_vc = df1[kkr_involved & mi_involved]['player_of_match'].value_counts()

plt.figure(figsize=(15,5))

g = sns.barplot(x=mi_vs_kkr_mom_vc.index, y = mi_vs_kkr_mom_vc)

plot_sns(g, ylabel = "MOM awards",title ="MOM winners in MI vs KKR matches", y_lim = mi_vs_kkr_mom_vc.max(), y_interval=1, face_grid=False)

winner_fielding_first = df1[df1.toss_decision == 'FIELD'].winner.value_counts().sort_index()

plt.figure(figsize=(8,5))



winner_fielding_first = winner_fielding_first.drop(labels=trivial_teams)

# print(winner_fielding_first)

g = sns.barplot(x=winner_fielding_first.index, y = winner_fielding_first, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "No of matches won",title ="Matches won fielding first", y_lim = winner_fielding_first.max(), 

         y_interval=10,  face_grid=False)

winner_batting_first = df1[df1.toss_decision == 'BAT'].winner.value_counts().sort_index()

# print(winner_batting_first)

plt.figure(figsize=(8,5))



# not using trivial_teams list here as Kochi team has not won even a single match batting first

winner_batting_first = winner_batting_first.drop(labels=[gl,rps,dch,pw])

g = sns.barplot(x=winner_batting_first.index, y = winner_batting_first, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "No of matches won",title ="Matches won batting first", y_lim = winner_batting_first.max(), 

         y_interval=10,  face_grid=False)

g = sns.catplot(x='winner', data = df1,kind="count", height=5, aspect = 3, hue="toss_decision")

plot_sns(g, ylabel = "No of matches",title ="Toss decisions in the matches won", y_lim = 65, y_interval=10)

match_win_col_name = "Matches Won"

toss_win_col_name = "Tosses Won"

df_combo = pd.DataFrame({match_win_col_name: match_winners, toss_win_col_name: toss_winners}).sort_values(['Matches Won'], ascending=False)

# we shud pass figsize as an argument instead of calling it separately as we are not operating on a single column, instead multiple columns

df_combo.plot(color=["SkyBlue","IndianRed"],kind='bar', grid=True, figsize=(10,5))

plot(xlabel = 'Teams', ylabel = 'No of matches', title = 'Match Wins vs Toss Wins', 

     y_lim = df_combo[[match_win_col_name, toss_win_col_name]].values.max())



toss_win_ratio = (toss_winners*100 // matches_per_team.sort_index())

plt.figure(figsize=(8,5))

g = sns.barplot(x=toss_win_ratio.index, y = toss_win_ratio, 

                palette = ipl_line_palette, linewidth=2.5, edgecolor=ipl_marker_palette)

plot_sns(g, ylabel = "%age of tosses won",title ="Toss %age of each team", y_lim = toss_win_ratio.max(), 

         y_interval=10,  face_grid=False)
def calc_points(row,team):

    val=0

    if row['winner'] == team:

        val = 1

    else:

        val = 0

    return val





def plot_team_journey(team):

    fig = plt.figure()

    chrt = 0

    for season in season_list:

        title = f'{team}\'s journey in IPL {season}'



        df_season = df1[df1['season'] == season]

        is_team_in_team1 = df_season['team1'] == team

        is_team_in_team2 = df_season['team2'] == team



        df_team = df_season[is_team_in_team1 | is_team_in_team2].sort_values(by='id')

        if len(df_team) != 0:



            df_team['outcome'] = df_team.apply(calc_points, args=(team,), axis = 1)

            chrt += 1 

            ax = fig.add_subplot(3,4, chrt)

            df_team.plot(x='date', y = 'outcome', kind = 'line', marker = 'D',markerfacecolor='b', rot = 45, ax = ax, 

                              figsize = (20,12), sharey = True, legend = None, color = 'IndianRed')

            plt.yticks([0,1], ['Lost','Won'])

            plt.xlabel(season)

            plt.tick_params(bottom=False,  labelbottom=False)



    plt.suptitle(title, fontsize=25, color='royalblue')

    plt.show()

    





plot_team_journey(mi)

plot_team_journey(kkr)

plot_team_journey(rcb)

plot_team_journey(csk)

plt.figure(figsize=(12,7))

df_win_by_runs = df1[df1['win_by_runs'] > 0]

g=sns.swarmplot(x='season', y='win_by_runs', data=df_win_by_runs)



plot_sns(g, ylabel = "Runs",title ="Victory margin by runs", y_lim = 150, 

         y_interval=10,  face_grid=False)

plt.figure(figsize=(12,7))

df_win_by_wickets = df1[df1['win_by_wickets'] > 0]

g=sns.swarmplot(x='season', y='win_by_wickets', data=df_win_by_wickets)

plot_sns(g, ylabel = "Wickets",title ="Victory margin by wickets", y_lim = 10, 

         y_interval=1,  face_grid=False)
plt.figure(figsize=(14,6))

sns.violinplot(y='season', x='toss_and_match_winners_same',data=df1)

plt.xlabel('')

plt.ylabel('')

plt.title('Won match as well as toss? ')

plt.show()

plt.figure(figsize=(14,6))

sns.violinplot(y='season', x='home_team_victorious',data=df1)

plt.xlabel('')

plt.ylabel('')

plt.title('Won match played on home ground? ')

plt.show()

plt.figure(figsize=(14,10))

sns.heatmap(pd.crosstab(df1.winner, df1.day),annot=True,cmap='YlGnBu', linewidths=1)

plt.title("Matches won by teams on different days of the week")

plt.xlabel('')

plt.ylabel('')

plt.show()

plt.figure(figsize=(14,8))

sns.heatmap(pd.crosstab(df1.season, df1.day, values = df1.win_by_runs, aggfunc = 'mean'), annot=True,cmap='YlGnBu', linewidths=3)

plt.title("Victory margin in runs")

plt.xlabel('')

plt.ylabel('')

plt.show()
plt.figure(figsize=(14,10))

sns.heatmap(pd.crosstab(df1.season, df1.day, values = df1.win_by_wickets, aggfunc = 'mean'), annot=True,cmap='YlGnBu', linewidths=3)

plt.title("Victory margin in wickets")

plt.xlabel('')

plt.ylabel('')

plt.show()
plt.figure(figsize=(18,8))

g = sns.swarmplot(x='day',y =df1[df1.win_by_wickets > 0].win_by_wickets, data =df1)

plot_sns(g, title='Day-wise distribution of wins by wickets',ylabel='Wickets',y_lim=10,y_interval=1, face_grid=False)



plt.figure(figsize=(18,8))

g = sns.swarmplot(x='day',y = df1[df1.win_by_runs > 0].win_by_runs, data =df1)

y_lim = df1.win_by_runs.max()

plot_sns(g, title='Day-wise distribution of wins by runs',ylabel='Runs',y_lim =y_lim,y_interval=10, face_grid=False)



plt.figure(figsize=(18,8))

g = sns.swarmplot(x = df1[~df1.winner.isin(trivial_teams)].winner, y=df1[df1.win_by_wickets > 0].win_by_wickets, data =df1)

plot_sns(g, title='Team-wise distribution of wins by wickets',ylabel='Wickets',y_lim=10,y_interval=2, face_grid=False)





plt.figure(figsize=(18,8))

g = sns.swarmplot(x = df1[~df1.winner.isin(trivial_teams)].winner, y=df1[df1.win_by_runs > 0].win_by_runs, data =df1)

y_lim = df1.win_by_runs.max()

plot_sns(g, title='Team-wise distribution of wins by runs',ylabel='Runs',y_lim =y_lim,y_interval=20, face_grid=False)

print('\n DL applied \n ')

print(df1.dl_applied.value_counts())

print(len(df1[df1.result == 'TIE']))
df2 = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')

df2.head()
print('='*20 + 'Exploratory Data Analysis' + '='*20)

print('='*20 + 'SHAPE' + '='*20)

print(df2.shape)

print('\n\n')

print('='*20 + 'INFO' + '='*20)

df2.info()

print('\n\n')

print('='*20 + 'DESCRIBE' + '='*20)

print(df2.describe())

print('\n\n')

print('='*20 + 'NA CHECK' + '='*20)

print(df2.isna().any())

print(df2.isna().sum())

print('\n\n')

print('='*20 + 'UNIQUE VALUES' + '='*20)

df2.apply(lambda x: pd.unique(x).tolist())


# I will only check those col for unique values where I'm not sure what possible values can the col take. For others it would be a sanity check, like wickets can't be more than 10 etc

print('\n over')

print(df2.over.unique())

print('\n ball')

print(df2.ball.unique())

print('\n batting_team')

print(df2.batting_team.unique())

print('\n bowling_team')

print(df2.bowling_team.unique())

print('\n is_super_over')

print(df2.is_super_over.unique())

print('\n penalty_runs')

print(df2.penalty_runs.unique())

print('\n dismissal_kind')

print(df2.dismissal_kind.unique())
# Looks like it is already sorted, but sorting again just in case...

df2 = df2.sort_values(by=['match_id','inning','over','ball'])



df2.replace({'Rising Pune Supergiant':'Rising Pune Supergiants', 'Delhi Daredevils': 'Delhi Capitals'}, inplace=True)



df2.fillna('', inplace=True)



# Removing leading/trailing whitespaces and changing to upper case

# TODO: There has to be a better way to do this!

df2.batting_team = df2.batting_team.str.strip().str.upper()

df2.bowling_team = df2.bowling_team.str.strip().str.upper()

df2.batsman = df2.batsman.str.strip().str.upper()

df2.non_striker = df2.non_striker.str.strip().str.upper()

df2.bowler = df2.bowler.str.strip().str.upper()

df2.player_dismissed = df2.player_dismissed.str.strip().str.upper()

df2.fielder = df2.fielder.str.strip().str.upper()



df2.head()
most_balls_faced_vc = df2.batsman.value_counts()

plt.figure(figsize=(18,5))



most_balls_faced_vc = most_balls_faced_vc[most_balls_faced_vc >= 2000]

g = sns.barplot(x=most_balls_faced_vc.index, y = most_balls_faced_vc)

plot_sns(g, ylabel = "Deliveries",title ="Most balls faced across all seasons", y_lim = most_balls_faced_vc.max(), y_interval=500, face_grid=False)







dot_balls_batsman = df2[ df2['batsman_runs'] == 0]['batsman'].value_counts()

dot_balls_batsman = dot_balls_batsman[dot_balls_batsman > 900]



plt.figure(figsize=(18,5))

g = sns.barplot(x = dot_balls_batsman.index, y = dot_balls_batsman)

plot_sns(g, title ="Batsmen who faced most dot balls", y_lim = dot_balls_batsman.max(), y_interval=200, face_grid=False)

# %%timeit

def calc_dot_ball_faced_streak(df):

    df['dotball_streak1'] = (df['batsman_runs'] == 0).cumsum()

    df['cumsum'] = np.nan

    df.loc[df.batsman_runs>0, 'cumsum'] = df['dotball_streak1']

    df['cumsum'] = df['cumsum'].fillna(method='ffill')

    df['cumsum'] = df['cumsum'].fillna(0)

    df['dotball_streak'] = df['dotball_streak1'] - df['cumsum']

    df.drop(['dotball_streak1', 'cumsum'], axis=1, inplace=True)

    return df



df3 = df2[['match_id','batsman','batsman_runs']]

df_dotball_streak = df3.groupby('batsman').apply(calc_dot_ball_faced_streak).sort_values(['batsman','match_id'])

df_dotball_streak = df_dotball_streak.sort_values('dotball_streak', ascending=False)

df_dotball_streak = df_dotball_streak.groupby('batsman').dotball_streak.max().sort_values(ascending=False)

df_dotball_streak = df_dotball_streak.reset_index()

df_dotball_streak = df_dotball_streak.nlargest(20,'dotball_streak')





plt.figure(figsize=(18,5))

g = sns.barplot(x = df_dotball_streak.batsman, y = df_dotball_streak.dotball_streak)

plot_sns(g, title ="Batsman who faced most consecutive dot-balls across matches", y_lim = int(df_dotball_streak.dotball_streak.max()), y_interval=5, face_grid=False)

# df3 = df2[['match_id','batsman','batsman_runs']]

# df_dotball_streak = df3.groupby(['match_id','batsman']).apply(calc_dot_ball_faced_streak).sort_values(['dotball_streak'])

# df_dotball_streak = df_dotball_streak.sort_values('dotball_streak', ascending=False)

# df_dotball_streak = df_dotball_streak.groupby('batsman').dotball_streak.max().sort_values(ascending=False)

# df_dotball_streak = df_dotball_streak.reset_index()

# df_dotball_streak = df_dotball_streak.nlargest(20,'dotball_streak')





# plt.figure(figsize=(18,5))

# g = sns.barplot(x = df_dotball_streak.batsman, y = df_dotball_streak.dotball_streak)

# plot_sns(g, title ="Batsman who faced most consecutive dot-balls in an inning", y_lim = int(df_dotball_streak.dotball_streak.max()), y_interval=3, face_grid=False)



# %load_ext line_profiler


def calc_dot_ball_bowled_streak(df):

    df['dotball_streak1'] = ( df['dotball_bowler'].to_numpy() == 1).cumsum()

    df['cumsum'] = np.nan

    df.loc[df['dotball_bowler'].to_numpy() == 0, 'cumsum'] = df['dotball_streak1']

    df['cumsum'] = df['cumsum'].fillna(method='ffill').fillna(0)

    df['dotball_streak'] = df['dotball_streak1'].to_numpy() - df['cumsum'].to_numpy()

    df.drop(['dotball_streak1', 'cumsum'], axis=1, inplace=True)

    return df



df2['dotball_bowler'] = ((df2['wide_runs'].to_numpy() == 0) & (df2['noball_runs'].to_numpy() == 0) & (df2['batsman_runs'].to_numpy() == 0)).astype(int)

df3 = df2[['match_id','bowler','dotball_bowler']]

# %lprun -f calc_dot_ball_bowled_streak df3.groupby('bowler').apply(calc_dot_ball_bowled_streak).sort_values(['bowler','match_id'])

df_dotball_streak = df3.groupby('bowler').apply(calc_dot_ball_bowled_streak).sort_values(['bowler','match_id'])

df_dotball_streak = df_dotball_streak[df_dotball_streak.dotball_streak>=0][['match_id','bowler','dotball_streak']]



df_dotball_streak = df_dotball_streak.sort_values('dotball_streak', ascending=False)

df_dotball_streak = df_dotball_streak.groupby('bowler').dotball_streak.max().sort_values(ascending=False)

df_dotball_streak = df_dotball_streak.reset_index()

df_dotball_streak = df_dotball_streak.nlargest(20,'dotball_streak')



plt.figure(figsize=(18,5))

g = sns.barplot(x = df_dotball_streak.bowler, y = df_dotball_streak.dotball_streak)

plot_sns(g, title ="Bowlers who bowled most consecutive dot-balls across matches", y_lim = int(df_dotball_streak.dotball_streak.max()), y_interval=2, face_grid=False)

# df2['dotball_bowler'] = ((df2['wide_runs'].values == 0) & (df2['noball_runs'].values == 0) & (df2['batsman_runs'].values == 0)).astype(int)

# df3 = df2[['match_id','bowler','dotball_bowler']]

# df_dotball_streak = df3.groupby(['match_id','bowler']).apply(calc_dot_ball_bowled_streak).sort_values(['dotball_streak'])



# df_dotball_streak = df_dotball_streak.sort_values('dotball_streak', ascending=False)

# df_dotball_streak = df_dotball_streak.groupby('bowler').dotball_streak.max().sort_values(ascending=False)

# df_dotball_streak = df_dotball_streak.reset_index()

# df_dotball_streak = df_dotball_streak.nlargest(20,'dotball_streak')

# df_dotball_streak

# plt.figure(figsize=(18,5))

# g = sns.barplot(x = df_dotball_streak.bowler, y = df_dotball_streak.dotball_streak)

# plot_sns(g, title ="Bowlers who bowled most consecutive dot-balls in an inning", y_lim = int(df_dotball_streak.dotball_streak.max()), y_interval=2, face_grid=False)

df3 = df2[['match_id','bowler','dotball_bowler']]



df_max_dotballs = df3.groupby(['match_id','bowler']).dotball_bowler.sum().sort_values(ascending=False)

df_max_dotballs = df_max_dotballs.reset_index(name="no")

df_max_dotballs = df_max_dotballs[df_max_dotballs.no > 17]

plt.figure(figsize=(18,5))

g = sns.barplot(x = df_max_dotballs.bowler, y = df_max_dotballs.no)

plot_sns(g, title ="Bowler with max dot balls in an innings", y_lim = df_max_dotballs.no.max(), y_interval=3, face_grid=False)



non_striker_batsman = df2.non_striker.value_counts()

non_striker_batsman

non_striker_batsman = non_striker_batsman[non_striker_batsman > 2000]



plt.figure(figsize=(18,5))

g = sns.barplot(x = non_striker_batsman.index, y = non_striker_batsman)

plot_sns(g, title ="Batsmen as non-striker for most balls", y_lim = non_striker_batsman.max(), y_interval=500, face_grid=False)

legbye_batsman = df2.groupby('batsman').legbye_runs.sum().sort_values(ascending=False)

legbye_batsman = legbye_batsman.reset_index(name="no")

legbye_batsman = legbye_batsman[legbye_batsman.no > 40]



plt.figure(figsize=(18,5))

g = sns.barplot(x = legbye_batsman.batsman, y = legbye_batsman.no)

plot_sns(g, title ="Batsmen with most runs from leg byes", y_lim = legbye_batsman.no.max(), y_interval=10, face_grid=False)

most_runs_vc = df2.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False)

vc = most_runs_vc[most_runs_vc > 2200]

plt.figure(figsize=(18,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Most runs scored", y_lim = vc.max(), y_interval=500, face_grid=False)

vc = df2['player_dismissed'].value_counts()

plt.figure(figsize=(18,5))

vc = vc[vc >= 80][1:]

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g,title ="Batsmen who has been dismissed most", y_lim = vc.max(), y_interval=20, face_grid=False)

most_bowler_faced = df2[df2.batsman == 'V KOHLI']['bowler'].value_counts()

# print(most_bowler_faced)

plt.figure(figsize=(18,5))



most_bowler_faced = most_bowler_faced[most_bowler_faced >= 50]

g = sns.barplot(x = most_bowler_faced.index, y = most_bowler_faced)

plot_sns(g,title ="Bowler who bowled most balls to Kohli", y_lim = most_bowler_faced.max(), y_interval=10, face_grid=False)



most_bowler_faced = df2[df2.batsman == 'CH GAYLE']['bowler'].value_counts()

# print(most_bowler_faced)

plt.figure(figsize=(18,5))



most_bowler_faced = most_bowler_faced[most_bowler_faced >= 40]

g = sns.barplot(x = most_bowler_faced.index, y = most_bowler_faced)

plot_sns(g,title ="Bowler who bowled most balls to Gayle", y_lim = most_bowler_faced.max(), y_interval=10, face_grid=False)



most_sixes_vc = df2[df2['batsman_runs'] == 6]['batsman'].value_counts()

# print(most_sixes_vc)

plt.figure(figsize=(18,5))



most_sixes_vc = most_sixes_vc[most_sixes_vc >= 80]

g = sns.barplot(x = most_sixes_vc.index, y = most_sixes_vc)

plot_sns(g, ylabel = "No of sixes",title ="Batsmen who hit most 6s", y_lim = most_sixes_vc.max(), y_interval=30, face_grid=False)



most_fours_vc = df2[df2['batsman_runs'] == 4]['batsman'].value_counts()

# print(most_fours_vc)

plt.figure(figsize=(18,5))



most_fours_vc = most_fours_vc[most_fours_vc >= 250]

g = sns.barplot(x = most_fours_vc.index, y = most_fours_vc)

plot_sns(g, ylabel = "No of fours",title ="Batsmen who hit most 4s", y_lim = most_fours_vc.max(), y_interval=50, face_grid=False)



most_balls_bowled_vc = df2.bowler.value_counts()

plt.figure(figsize=(18,5))



most_balls_bowled_vc = most_balls_bowled_vc[most_balls_bowled_vc >= 1700]

g = sns.barplot(x = most_balls_bowled_vc.index, y = most_balls_bowled_vc)

plot_sns(g, ylabel = "Deliveries",title ="Bowlers who bowled most deliveries", y_lim = most_balls_bowled_vc.max(), y_interval=500, face_grid=False)



most_runs_given_vc = df2.groupby('bowler')[['noball_runs','batsman_runs','wide_runs']].apply(lambda x : x.astype(int).sum())

vc = most_runs_given_vc.sum(axis=1).sort_values(ascending=False)

vc = vc[vc > 2300]

# print(vc)

plt.figure(figsize=(18,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Most runs conceded", y_lim = vc.max(), y_interval=500, face_grid=False)

max_wide_and_no_balls_bowlers_vc = df2[(df2['wide_runs'] != 0) | (df2['noball_runs'] != 0)]['bowler'].value_counts()

plt.figure(figsize=(18,5))

vc =  max_wide_and_no_balls_bowlers_vc

vc = vc[vc > 70]

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Bowlers who bowled most wide and no balls", y_lim = vc.max(), y_interval=20, face_grid=False)

dismissal_kind_vc = df2['dismissal_kind'].value_counts()

# print(dismissal_kind_vc)

dismissals_not_by_bowler = ['run out','retired hurt', 'obstructing the field','']

dismissals_by_bowler_vc = dismissal_kind_vc.drop(labels = dismissals_not_by_bowler)

plt.figure(figsize=(18,5))

# print(dismissals_by_bowler_vc.index)

most_wickets_vc = df2[df2['dismissal_kind'].isin(dismissals_by_bowler_vc.index)]['bowler'].value_counts()

# print(most_wickets_vc.head(20))

most_wickets_vc = most_wickets_vc[most_wickets_vc >= 100]

g = sns.barplot(x = most_wickets_vc.index, y = most_wickets_vc)

plot_sns(g, ylabel = "Wickets",title ="Highest wicket takers across all seasons", y_lim = most_wickets_vc.max(), y_interval=20, face_grid=False)



dismissal_kind_vc = df2['dismissal_kind'].value_counts()

plt.figure(figsize=(10,5))

g = sns.barplot(x = dismissal_kind_vc[1:7].index, y = dismissal_kind_vc[1:7])

plot_sns(g, title ="Mode of dismissals", y_lim = dismissal_kind_vc[1:7].max(), y_interval=500, face_grid=False)



plt.figure(figsize=(16,6))



star_batsmen = ['V KOHLI','MS DHONI','CH GAYLE','AB DE VILLIERS','SK RAINA','RG SHARMA']

dismiss_type = ~df2.dismissal_kind.isin(dismissals_not_by_bowler)

dismiss_type = ~df2.dismissal_kind.isin(['caught'])

dismissals = df2[df2.player_dismissed.isin(star_batsmen) & dismiss_type]

dismissals = dismissals[['player_dismissed','dismissal_kind']]

dismissals = dismissals.groupby(['player_dismissed']).dismissal_kind.value_counts()

dismissals = dismissals.reset_index(name='no')



g = sns.pointplot(x=dismissals.player_dismissed, y = dismissals.no, hue=dismissals.dismissal_kind, dodge=False)

plot_sns(g, ylabel = "No of dismissals",title ="Ways of getting dismissed (excludes 'caught')", y_lim = dismissals.no.max(), y_interval=2, face_grid=False)

# dismissals





fielder_vc = df2.fielder.value_counts()

plt.figure(figsize=(18,5))

fielder_vc = fielder_vc[1:]

# print(fielder_vc.head(20))

fielder_vc = fielder_vc[fielder_vc >= 40]

g = sns.barplot(x = fielder_vc.index, y = fielder_vc)

plot_sns(g, ylabel = "Dismissals",title ="Fielders (including wicket-keepers) who affected most dismissals", y_lim = fielder_vc.max(), y_interval=25, face_grid=False)

wicket_keepers = ['MS Dhoni','KD Karthik','RV Uthappa','PA Patel','NV Ojha','WP Saha','AC Gilchrist','KC Sangakkara','SV Samson']

wicket_keepers = [x.upper() for x in wicket_keepers]

wicket_keepers_vc = df2[df2['fielder'].isin(wicket_keepers)].fielder.value_counts()

plt.figure(figsize=(12,5))

# print(wicket_keepers_vc.head(20))

g = sns.barplot(x = wicket_keepers_vc.index, y = wicket_keepers_vc)

plot_sns(g, ylabel = "Dismissals",title ="Wicket-keepers who affected most dismissals (including run-outs)", y_lim = wicket_keepers_vc.max(), y_interval=25, face_grid=False)



fielder_exlcuding_wk_vc = fielder_vc.drop(labels=wicket_keepers)

plt.figure(figsize=(15,5))

# print(fielder_exlcuding_wk_vc.head(20))

fielder_exlcuding_wk_vc = fielder_exlcuding_wk_vc[fielder_vc > 10]

g = sns.barplot(x = fielder_exlcuding_wk_vc.index, y = fielder_exlcuding_wk_vc)

plot_sns(g, ylabel = "Dismissals",title ="Fielders (excluding wicket-keepers) who affected most dismissals", y_lim = fielder_exlcuding_wk_vc.max(), y_interval=25, face_grid=False)

max_wide_balls_bowlers_vc = df2[df2['wide_runs'] != 0]['bowler'].value_counts()

plt.figure(figsize=(18,5))

vc =  max_wide_balls_bowlers_vc

# print(max_noballs_bowlers_vc.head(20))

vc = vc[vc > 50]

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Bowlers who bowled most wide balls", y_lim = vc.max(), y_interval=20, face_grid=False)

max_noballs_bowlers_vc = df2[df2['noball_runs'] != 0]['bowler'].value_counts()

plt.figure(figsize=(18,5))

# print(max_noballs_bowlers_vc.head(20))

max_noballs_bowlers_vc = max_noballs_bowlers_vc[max_noballs_bowlers_vc > 7]

g = sns.barplot(x = max_noballs_bowlers_vc.index, y = max_noballs_bowlers_vc)

plot_sns(g, title ="Bowlers who bowled most no-balls", y_lim = max_noballs_bowlers_vc.max(), y_interval=5, face_grid=False)

noball_runs_exc_freehit_vc = df2[df2['noball_runs'] != 0].groupby('bowler')['total_runs'].sum().sort_values(ascending=False)

noball_runs_exc_freehit_vc = noball_runs_exc_freehit_vc[noball_runs_exc_freehit_vc > 25]



plt.figure(figsize=(18,5))

g = sns.barplot(x = noball_runs_exc_freehit_vc.index, y = noball_runs_exc_freehit_vc)

plot_sns(g, title ="Most runs given off noballs (excludes free hits)", y_lim = noball_runs_exc_freehit_vc.max(), y_interval=10, face_grid=False)

dot_balls_vc = df2[ (df2['wide_runs'] == 0) & (df2['noball_runs'] == 0) & (df2['batsman_runs'] == 0)]['bowler'].value_counts()

# print(dot_balls_vc)

dot_balls_vc = dot_balls_vc[dot_balls_vc > 800]



plt.figure(figsize=(18,5))

g = sns.barplot(x = dot_balls_vc.index, y = dot_balls_vc)

plot_sns(g, title ="Bowlers with max dot balls", y_lim = dot_balls_vc.max(), y_interval=200, face_grid=False)

gayle_most_runs_vc = df2[df2.batsman == 'CH GAYLE'].groupby('bowler')['batsman_runs'].sum().sort_values(ascending=False)

vc = gayle_most_runs_vc[gayle_most_runs_vc > 60]

plt.figure(figsize=(18,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Bowlers against which Gayle scored most runs", y_lim = vc.max(), y_interval=20, face_grid=False)

gayle_sixes_vc = df2[(df2.batsman == 'CH GAYLE') & (df2.batsman_runs == 6)].bowler.value_counts()

# print(gayle_sixes_vc)

gayle_sixes_vc = gayle_sixes_vc[gayle_sixes_vc > 4]



plt.figure(figsize=(18,5))

g = sns.barplot(x = gayle_sixes_vc.index, y = gayle_sixes_vc)

plot_sns(g, title ="Bowlers who were hit for most 6s by Gayle", y_lim = gayle_sixes_vc.max(), y_interval=2, face_grid=False)

gayle_fours_vc = df2[(df2.batsman == 'CH GAYLE') & (df2.batsman_runs == 4)].bowler.value_counts()

gayle_fours_vc = gayle_fours_vc[gayle_fours_vc > 5]



plt.figure(figsize=(18,5))

g = sns.barplot(x = gayle_fours_vc.index, y = gayle_fours_vc)

plot_sns(g, title ="Bowlers who were hit for most 4s by Gayle", y_lim = gayle_fours_vc.max(), y_interval=2, face_grid=False)

raina_most_runs_vc = df2[df2.batsman == 'SK RAINA'].groupby('bowler')['batsman_runs'].sum().sort_values(ascending=False)

vc = raina_most_runs_vc

vc = vc[vc > 70]



plt.figure(figsize=(18,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Bowlers against which Raina scored most runs", y_lim = vc.max(), y_interval=20, face_grid=False)

kohli_most_runs_vc = df2[df2.batsman == 'V KOHLI'].groupby('bowler')['batsman_runs'].sum().sort_values(ascending=False)

vc = kohli_most_runs_vc

vc = vc[vc > 70]

plt.figure(figsize=(18,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Bowlers against which Kohli scored most runs", y_lim = vc.max(), y_interval=20, face_grid=False)

abd_sixes_vc = df2[(df2.batsman == 'AB DE VILLIERS') & (df2.batsman_runs == 6)].bowler.value_counts()

abd_sixes_vc = abd_sixes_vc[abd_sixes_vc > 3]



plt.figure(figsize=(12,5))

g = sns.barplot(x = abd_sixes_vc.index, y = abd_sixes_vc)

plot_sns(g, title ="Bowlers who were hit for most 6s by ABD", y_lim = abd_sixes_vc.max(), y_interval=2, face_grid=False)

abd_fours_vc = df2[(df2.batsman == 'AB DE VILLIERS') & (df2.batsman_runs == 4)].bowler.value_counts()

abd_fours_vc = abd_fours_vc[abd_fours_vc > 5]



plt.figure(figsize=(15,5))

g = sns.barplot(x = abd_fours_vc.index, y = abd_fours_vc)

plot_sns(g, title ="Bowlers who were hit for most 4s by ABD", y_lim = abd_fours_vc.max(), y_interval=2, face_grid=False)

msd_sixes_vc = df2[(df2.batsman == 'MS DHONI') & (df2.batsman_runs == 6)].bowler.value_counts()

msd_sixes_vc = msd_sixes_vc[msd_sixes_vc > 3]



plt.figure(figsize=(15,5))

g = sns.barplot(x = msd_sixes_vc.index, y = msd_sixes_vc)

plot_sns(g, title ="Bowlers who were hit for most 6s by MSD", y_lim = msd_sixes_vc.max(), y_interval=2, face_grid=False)

msd_fours_vc = df2[(df2.batsman == 'MS DHONI') & (df2.batsman_runs == 4)].bowler.value_counts()

msd_fours_vc = msd_fours_vc[msd_fours_vc > 4]



plt.figure(figsize=(15,5))

g = sns.barplot(x = msd_fours_vc.index, y = msd_fours_vc)

plot_sns(g, title ="Bowlers who were hit for most 4s by MSD", y_lim = msd_fours_vc.max(), y_interval=2, face_grid=False)

raina_sixes_vc = df2[(df2.batsman == 'SK RAINA') & (df2.batsman_runs == 6)].bowler.value_counts()

raina_sixes_vc = raina_sixes_vc[raina_sixes_vc > 3]



plt.figure(figsize=(12,5))

g = sns.barplot(x = raina_sixes_vc.index, y = raina_sixes_vc)

plot_sns(g, title ="Bowlers who were hit for most 6s by Raina", y_lim = raina_sixes_vc.max(), y_interval=2, face_grid=False)

kohli_sixes_vc = df2[(df2.batsman == 'V KOHLI') & (df2.batsman_runs == 6)].bowler.value_counts()

kohli_sixes_vc = kohli_sixes_vc[kohli_sixes_vc > 2]



plt.figure(figsize=(15,5))

g = sns.barplot(x = kohli_sixes_vc.index, y = kohli_sixes_vc)

plot_sns(g, title ="Bowlers who were hit for most 6s by Kohli", y_lim = kohli_sixes_vc.max(), y_interval=2, face_grid=False)

rohit_sixes_vc = df2[(df2.batsman == 'RG SHARMA') & (df2.batsman_runs == 6)].bowler.value_counts()

rohit_sixes_vc = rohit_sixes_vc[rohit_sixes_vc > 3]



plt.figure(figsize=(18,5))

g = sns.barplot(x = rohit_sixes_vc.index, y = rohit_sixes_vc)

plot_sns(g, title ="Bowlers who were hit for most 6s by Rohit", y_lim = rohit_sixes_vc.max(), y_interval=2, face_grid=False)

gayle_out_bowler_vc = df2[(df2.player_dismissed == 'CH GAYLE') & (~df2.dismissal_kind.isin(dismissals_not_by_bowler) )].bowler.value_counts()

gayle_out_bowler_vc = gayle_out_bowler_vc[gayle_out_bowler_vc > 2]



plt.figure(figsize=(12,5))

g = sns.barplot(x = gayle_out_bowler_vc.index, y = gayle_out_bowler_vc)

plot_sns(g, title ="Bowlers who dismissed Gayle most times", y_lim = gayle_out_bowler_vc.max(), y_interval=2, face_grid=False)

ABD_out_bowler_vc = df2[(df2.player_dismissed == 'AB DE VILLIERS') & (~df2.dismissal_kind.isin(dismissals_not_by_bowler) )].bowler.value_counts()

ABD_out_bowler_vc = ABD_out_bowler_vc[ABD_out_bowler_vc > 2]



plt.figure(figsize=(10,4))

g = sns.barplot(x = ABD_out_bowler_vc.index, y = ABD_out_bowler_vc)

plot_sns(g, title ="Bowlers who dismissed ABD most times", y_lim = ABD_out_bowler_vc.max(), y_interval=2, face_grid=False)

MSD_out_bowler_vc = df2[(df2.player_dismissed == 'MS DHONI') & (~df2.dismissal_kind.isin(dismissals_not_by_bowler) )].bowler.value_counts()

MSD_out_bowler_vc = MSD_out_bowler_vc[MSD_out_bowler_vc > 2]



plt.figure(figsize=(10,4))

g = sns.barplot(x = MSD_out_bowler_vc.index, y = MSD_out_bowler_vc)

plot_sns(g, title ="Bowlers who dismissed MSD most times", y_lim = MSD_out_bowler_vc.max(), y_interval=2, face_grid=False)

kohli_out_bowler_vc = df2[(df2.player_dismissed == 'V KOHLI') & (~df2.dismissal_kind.isin(dismissals_not_by_bowler) )].bowler.value_counts()

kohli_out_bowler_vc = kohli_out_bowler_vc[kohli_out_bowler_vc > 2]



plt.figure(figsize=(12,5))

g = sns.barplot(x = kohli_out_bowler_vc.index, y = kohli_out_bowler_vc)

plot_sns(g, title ="Bowlers who dismissed Kohli most times", y_lim = kohli_out_bowler_vc.max(), y_interval=2, face_grid=False)

run_out_fielder_vc = df2[df2.dismissal_kind == 'run out'].fielder.value_counts()

vc = run_out_fielder_vc[run_out_fielder_vc > 8]

plt.figure(figsize=(15,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Fielders affecting most run outs", y_lim = vc.max(), y_interval=5, face_grid=False)

run_out_batsman_vc = df2[df2.dismissal_kind == 'run out'].player_dismissed.value_counts()

vc = run_out_batsman_vc[run_out_batsman_vc > 8]

# print(vc)

plt.figure(figsize=(15,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Batsman who were dismissed most via run outs", y_lim = vc.max(), y_interval=5, face_grid=False)

batsman_run_out = df2[(df2.dismissal_kind == 'run out') & (df2.batsman == df2.player_dismissed)].non_striker.value_counts()

nonstriker_run_out = df2[(df2.dismissal_kind == 'run out') & (df2.non_striker == df2.player_dismissed)].batsman.value_counts()



vc = (batsman_run_out + nonstriker_run_out).sort_values(ascending=False)

vc = vc[vc > 10]

plt.figure(figsize=(18,5))

g = sns.barplot(x = vc.index, y = vc)

plot_sns(g, title ="Batsman who ran others out most", y_lim = 25, y_interval=5, face_grid=False)

plt.figure(figsize=(12,5))

bin_edges = np.arange(-0.5,7.5)

plt.hist(df2.batsman_runs,bins=bin_edges)

plt.ylabel('Runs')

plt.title('Comparison of dots, singles, doubles, triples, 4s and 6s')

plt.show()
plt.figure(figsize=(12,5))

df_dhoni_kohli = df2[(df2.batsman == 'MS DHONI') | (df2.batsman == 'V KOHLI')][['batsman','batsman_runs']]

bin_edges = np.arange(-0.5,7.5)

plt.hist(df_dhoni_kohli[df_dhoni_kohli.batsman == 'MS DHONI'].batsman_runs, bins = bin_edges, alpha = 0.5)

plt.hist(df_dhoni_kohli[df_dhoni_kohli.batsman == 'V KOHLI'].batsman_runs, bins = bin_edges, alpha = 0.3)

plt.legend(('MS DHONI','V KOHLI'))

plt.ylabel('Runs')

plt.title('DHONI v KOHLI: Comparison of dots, singles, doubles, triples, 4s and 6s')

plt.show()
def ecdf(data):

    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n

    n = len(data)



    # x-data for the ECDF: x

    x = np.sort(data)



    # y-data for the ECDF: y

    y = np.arange(1, n+1) / n



    return x, y





def pearson_r(x, y):

    """Compute Pearson correlation coefficient between two arrays."""

    # Compute correlation matrix: 

    corr_mat = np.corrcoef(x,y)



    # Return entry [0,1]

    return corr_mat[0,1]
print(pearson_r(df1.season , df1.win_by_runs))

print(pearson_r(df1.season , df1.win_by_wickets))

print(pearson_r(df1.home_team_victorious , df1.win_by_wickets))

print(pearson_r(df1.home_team_victorious , df1.win_by_runs))

print(pearson_r(df1.toss_and_match_winners_same , df1.win_by_wickets))

print(pearson_r(df1.toss_and_match_winners_same , df1.win_by_runs))
plt.figure(figsize=(12,5))

x_vers , y_vers = ecdf(df2.batsman_runs)

plt.plot(x_vers, y_vers, marker = '.' , linestyle = 'none')

plt.xlabel('Batsman Runs')

plt.ylabel('ECDF')

plt.title('Batsman Runs-Comparison of dots, singles, doubles, triples, 4s and 6s')

plt.show()
plt.figure(figsize=(12,5))

x_vers , y_vers = ecdf(df2.total_runs)

plt.plot(x_vers, y_vers, marker = '.' , linestyle = 'none')

plt.xlabel('Total Runs')

plt.ylabel('ECDF')

plt.title('Total runs off a ball')

plt.show()
plt.figure(figsize=(12,5))

x_vers , y_vers = ecdf(df1.win_by_runs)

plt.plot(x_vers, y_vers, marker = '.' , linestyle = 'none')

plt.xlabel('Victory margin in runs')

plt.ylabel('ECDF')

plt.margins(0.1)

plt.show()
plt.figure(figsize=(12,5))

x_vers , y_vers = ecdf(df1.win_by_wickets)

plt.plot(x_vers, y_vers, marker = '.' , linestyle = 'none')

plt.xlabel('Victory margin in wickets')

plt.ylabel('ECDF')

plt.margins(0.02)

plt.show()
df_corr = df2[df2.batsman == 'MS DHONI'].groupby('match_id').batsman_runs.sum().to_frame(name = 'score').reset_index()

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['score','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['RPS','CSK'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

print(df_corr.head(10))

print(f'Correlation between MSD\'s scores and team\s victory is POSITIVE as expected: { pearson_r(df_corr.score, df_corr.victorious)}' )

plt.plot(df_corr.score, df_corr.victorious, marker='.', linestyle='none')

plt.xlabel('MSD\'s score')

plt.ylabel('Match result')

plt.yticks((0,1), ['Lost','Won']);
df_corr = df2[df2.batsman == 'MS DHONI'].groupby('match_id').batsman_runs.sum().to_frame(name = 'score').reset_index()

df_corr = df_corr[df_corr.score >= 50]

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['score','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['RPS','CSK'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

df_corr

print(f'Correlation between MSD\'s 50+ scores and team\s victory turns NEGATIVE: { pearson_r(df_corr.score, df_corr.victorious)}' )

plt.plot(df_corr.score, df_corr.victorious, marker='.', linestyle='none')

plt.xlabel('MSD\'s score')

plt.ylabel('Match result')

plt.yticks((0,1), ['Lost','Won']);
df_corr = df2[df2.batsman == 'V KOHLI'].groupby('match_id').batsman_runs.sum().to_frame(name = 'score').reset_index()

df_corr = df_corr[df_corr.score >= 50]

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['score','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['RCB'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

# df_corr

print(f'Correlation between KOHLI\'s 50+ scores and team\s victory is POSITIVE: { pearson_r(df_corr.score, df_corr.victorious)}' )

plt.plot(df_corr.score, df_corr.victorious, marker='.', linestyle='none')

plt.xlabel('KOHLI\'s score')

plt.ylabel('Match result')

plt.yticks((0,1), ['Lost','Won']);
df_corr = df2[df2.batsman == 'CH GAYLE'].groupby('match_id').batsman_runs.sum().to_frame(name = 'score').reset_index()

df_corr = df_corr[df_corr.score >= 50]

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['score','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['RCB'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

# df_corr

print(f'Correlation between GAYLE\'s 50+ scores and team\s victory is POSITIVE: { pearson_r(df_corr.score, df_corr.victorious)}' )

plt.plot(df_corr.score, df_corr.victorious, marker='.', linestyle='none')

plt.xlabel('GAYLE\'s score')

plt.ylabel('Match result')

plt.yticks((0,1), ['Lost','Won']);
df_corr = df2[df2.batsman == 'AB DE VILLIERS'].groupby('match_id').batsman_runs.sum().to_frame(name = 'score').reset_index()

df_corr = df_corr[df_corr.score >= 50]

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['score','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['RCB'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

# df_corr

print(f'Correlation between ABD\'s 50+ scores and team\s victory is weak positive: { pearson_r(df_corr.score, df_corr.victorious)}' )

plt.plot(df_corr.score, df_corr.victorious, marker='.', linestyle='none')

plt.xlabel('ABD\'s score')

plt.ylabel('Match result')

plt.yticks((0,1), ['Lost','Won']);
df_corr = df2[df2.batsman == 'SK RAINA'].groupby('match_id').batsman_runs.sum().to_frame(name = 'score').reset_index()

df_corr = df_corr[df_corr.score >= 50]

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['score','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['GL','CSK'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

# df_corr

print(f'Correlation between RAINA\'s 50+ scores and team\s victory is NEGLIGIBLE: { pearson_r(df_corr.score, df_corr.victorious)}' )

plt.plot(df_corr.score, df_corr.victorious, marker='.', linestyle='none')

plt.xlabel('RAINA\'s score')

plt.ylabel('Match result')

plt.yticks((0,1), ['Lost','Won']);
df_corr = df2[df2.batsman == 'RG SHARMA'].groupby('match_id').batsman_runs.sum().to_frame(name = 'score').reset_index()

df_corr = df_corr[df_corr.score >= 50]

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['score','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['DCH','MI'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

# df_corr

print(f'Correlation between ROHIT\'s 50+ scores and team\s victory is weak positive: { pearson_r(df_corr.score, df_corr.victorious)}' )

plt.plot(df_corr.score, df_corr.victorious, marker='.', linestyle='none')

plt.xlabel('ROHIT\'s score')

plt.ylabel('Match result')

plt.yticks((0,1), ['Lost','Won']);
df_corr = df2[~df2.dismissal_kind.isin(dismissals_not_by_bowler)]

df_corr = df_corr.groupby(['match_id']).bowler.value_counts()

df_corr = df_corr.reset_index(name='wickets')

df_corr = df_corr[df_corr.bowler == 'SL MALINGA']

df_corr = df_corr[df_corr.wickets > 3]

df_corr = pd.merge(df_corr, df1, left_on='match_id', right_on='id')

df_corr = df_corr[['wickets','winner']]

df_corr['victorious'] = (df_corr.winner.isin(['MI'])).astype(int)

df_corr.drop(columns='winner', inplace=True)

print(df_corr)

print(f'Correlation between MALINGA\'s 4+ wickets and team\s victory is weak positive: { pearson_r(df_corr.wickets, df_corr.victorious)}' )
