import numpy as np                                                 # Implemennts milti-dimensional array and matrices
import pandas as pd                                                # For data manipulation and analysis
import pandas_profiling
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics
%matplotlib inline
sns.set()

from subprocess import check_output
 # Importing dataset using pd.read_csv
match_data = pd.read_csv("../input/ipl/matches.csv")
delivery_data = pd.read_csv("../input/ipl/deliveries.csv")
match_data.shape                                           # This will print the number of rows and comlumns of the Data Frame
delivery_data.shape
match_data.columns                                         # This will print the names of all columns.
delivery_data.columns                                      # This will print the names of all columns.
match_data.head(2)
match_data.tail(2)
delivery_data.head(6)
delivery_data.tail(3)
match_data.info()
match_data.isnull().sum()
delivery_data.info()
profile = pandas_profiling.ProfileReport(match_data)
profile.to_file(output_file="matchdata_before_preprocessing.html")
profile = pandas_profiling.ProfileReport(delivery_data)
profile.to_file(output_file="delivery_data_before_preprocessing.html")
match_data.drop(["umpire3"],axis=1,inplace=True)      #drop umpire3 column and replace null values
delivery_data.fillna("playing",inplace=True)          #player_dismissed , dismissal_kind, fielder-in these column have null values because player is playing on these balls so we replce these null value with playing keyword.
match_data.venue[match_data.city.isnull()]
match_data.city.fillna("Dubai",inplace=True)          #repalce null values in city
match_data.date = pd.to_datetime(match_data.date)    #Convert Date in datetime object.
type(match_data.date[2])
match_data.fillna({"winner":"no result","player_of_match":"no result"},inplace=True)
match_data.umpire1.fillna(match_data.umpire1.mode()[0],inplace=True)             #repalce null values with mode
match_data.umpire2.fillna(match_data.umpire2.mode()[0],inplace=True)              #repalce null values with mode
match_data.info()
match_data.team1.unique()
iplmatches = match_data.copy()
ipldelivery =delivery_data.copy()
iplmatches.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
       'Rising Pune Supergiant', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
        ['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace=True)
ipldelivery.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
       'Rising Pune Supergiant', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
        ['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace=True)
iplmatches.tail(3)
profile = pandas_profiling.ProfileReport(iplmatches)
profile.to_file(output_file="iplmatches_after_preprocessing.html")
profile = pandas_profiling.ProfileReport(ipldelivery)
profile.to_file(output_file="ipldelivery_after_preprocessing.html")
iplmatches.shape[0]
df_total_match_played_by_teams = pd.DataFrame()
df_total_match_played_by_teams["team"] = pd.concat([iplmatches["team1"], iplmatches["team2"]], ignore_index=True, sort=False)
df_total_match_played_by_teams.head()
df_total_match_played_by_teams.groupby("team")["team"].count()
sns.countplot(x="team", data=df_total_match_played_by_teams).set_title('Total match played by each IPL team.')
iplmatches.groupby(["season"])["id"].count().sort_values(ascending=False)
plt.subplots(figsize=(10,6))
sns.countplot(x='season',data=iplmatches,palette=sns.color_palette('Paired'))  
plt.title("Number of IPL Matches Played in Each Season")
plt.show()
iplmatches.city.unique()
iplmatches.replace("Bangalore","Bengaluru",inplace=True)
iplmatches.city.unique()
iplmatches.groupby(["city"])["id"].count().sort_values(ascending=True).plot(kind='barh',figsize=(8,10))
plt.xlabel("Match Count")
plt.title("Cities and IPL Matches")
iplmatches.groupby(["city"])["id"].count().sort_values(ascending=False)     
iplmatches.groupby(['winner'])['id'].count().sort_values(ascending=False)
sns.countplot(x='winner', data=iplmatches).set_title('Most winning team of IPL')
iplmatches.groupby(['player_of_match'])['player_of_match'].count().sort_values(ascending=False)
df=iplmatches.iloc[[iplmatches['win_by_runs'].idxmax()]]
df[['season','city','team1','team2','winner','win_by_runs']]
df=iplmatches.iloc[[iplmatches['win_by_wickets'].idxmax()]]
df[['season','city','team1','team2','winner','win_by_wickets']]
(iplmatches.groupby(['toss_decision'])["toss_decision"].count()/iplmatches.toss_decision.count()*100).plot(kind="bar")
plt.subplots(figsize=(10,6))
sns.countplot(x='season',hue='toss_decision',data=iplmatches)
plt.title("Toss decision taken across the IPL season")
plt.show()
plt.subplots(figsize=(10,6))
ax=iplmatches['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('Paired',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.title("Most Toss Winner of IPL")
plt.show()
print('Winners By Years')
for i in range(2008,2018):
    df=((iplmatches[iplmatches['season']==i]).iloc[-1])
    print(df[[1,10]].values)
matchesdata = iplmatches.copy()
matchesdata.groupby("team1")['id'].count()     #GL=14,RPS=15,KTK=7,PW=20    mean=53 , median=62
teams = ['GL','RPS','KTK','PW']
for team in teams:
     matchesdata.drop(matchesdata[matchesdata.team1.str.contains(team)]['id'].index,axis=0,inplace=True)
teams = ['GL','RPS','KTK','PW']
for team in teams:
     matchesdata.drop(matchesdata[matchesdata.team2.str.contains(team)]['id'].index,axis=0,inplace=True)
matchesdata[matchesdata.toss_winner==matchesdata.team1][matchesdata.toss_decision=='bat'].groupby('team1')['toss_decision'].count().plot(kind='bar', figsize=(15, 7), color='orange')
matchesdata[matchesdata.toss_winner==matchesdata.team1][matchesdata.toss_decision=='bat'][matchesdata.winner==matchesdata.team1].groupby('team1')['toss_decision'].count().plot(kind='bar', figsize=(15, 7), color='grey')

plt.xlabel('Teams')
plt.ylabel('Count')
plt.title('Stacked Bar Chart showing the Toss Decision as Bat and Winning the Match')
plt.legend(labels=('Total Toss Winner with Bat', 'TossWin_Choose_Bat_Wining_Match'))
matchesdata[matchesdata.toss_winner==matchesdata.team2][matchesdata.toss_decision=='field'].groupby('team2')['toss_decision'].count().plot(kind='bar', figsize=(15, 7), color='orange')
matchesdata[matchesdata.toss_winner==matchesdata.team2][matchesdata.toss_decision=='field'][matchesdata.winner==matchesdata.team2].groupby('team2')['toss_decision'].count().plot(kind='bar', figsize=(15, 7), color='gray', fontsize=13)

plt.xlabel('Teams')
plt.ylabel('Count')
plt.title('Stacked Bar Chart showing the Toss Decision as Field and Winning the Match')
plt.legend(labels=('Total Toss Winner with Field', 'TossWin_Choose_Field_Wining_Match'))
df=iplmatches[iplmatches['toss_winner']==iplmatches['winner']]
slices=[len(df),(696-len(df))]
labels=['yes','no']
plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0.08),autopct='%1.1f%%',colors=['b','yellow'])
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.title("Toss Win vs Match Win",fontsize=20)
plt.show()
plt.subplots(figsize=(8,15))
iplmatches.groupby('venue')['venue'].count().sort_values(ascending=True).plot(kind='barh',color=sns.color_palette('inferno',40))
plt.xlabel('Count')
plt.ylabel('Grounds')
plt.show() 
plt.figure(figsize=(10,5))
umpire=pd.concat([iplmatches['umpire1'],iplmatches['umpire2']]).value_counts().sort_values(ascending=False)
umpire=umpire[:7].plot(kind='barh',color=sns.color_palette('colorblind',50))
plt.title('Favorite umpire')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Name of the Umpire', fontsize=15)
plt.show()
x, y = 2008, 2018
while x < y:
    wins_percity = matchesdata[matchesdata['season'] == x].groupby(['winner', 'city'])['id'].count().unstack()
    plot = wins_percity.plot(kind='bar', stacked=True, title="Team wins in different cities\nSeason "+str(x), figsize=(7, 5))
    sns.set_palette("Paired", len(matchesdata['city'].unique()))
    plot.set_xlabel("Teams")
    plot.set_ylabel("No of wins")
    plot.legend(loc='best', prop={'size':8})
    x+=1
batsmen = iplmatches[['id','season']].merge(ipldelivery, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
#merging the matches and delivery dataframe by referencing the id and match_id columns respectively
season=batsmen.groupby(['season'])['total_runs'].sum().reset_index()
season.set_index('season').plot(marker='o',color="b")
plt.gcf().set_size_inches(10,6)
plt.title('Total Runs Across the Seasons')
plt.show()
Season_boundaries=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()
a=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()
Season_boundaries=Season_boundaries.merge(a,left_on='season',right_on='season',how='left')
Season_boundaries=Season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})
Season_boundaries.set_index('season')[['6"s','4"s']].plot(marker='o',color=("r","b"))
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
batsman_runsperseason = batsmen.groupby(['season', 'batting_team', 'batsman'])['batsman_runs'].sum().reset_index()
batsman_runsperseason = batsman_runsperseason.groupby(['season', 'batsman'])['batsman_runs'].sum().unstack().T
batsman_runsperseason['Total'] = batsman_runsperseason.sum(axis=1) #add total column to find batsman with the highest runs
batsman_runsperseason = batsman_runsperseason.sort_values(by = 'Total', ascending = False).drop('Total', 1)
ax = batsman_runsperseason[:5].T.plot(figsize=(10,8),title="Top 5 Batsman Performance")

ipldelivery["wickets"] = (ipldelivery.player_dismissed != "playing") & (ipldelivery.player_dismissed != "retired hurt")
ipldelivery.groupby(['bowler'])['wickets'].sum().sort_values(ascending = False)[:10]
bowlers_wickets = ipldelivery.groupby(['bowler'])['wickets'].sum()
bowlers_wickets.sort_values(ascending = False, inplace = True)
bowlers_wickets[:10].plot(x='bowler', kind = 'bar', colormap = 'summer',figsize=(10,6))
plt.title("Top 10 Bowlers")
plt.ylabel("Total Wickets in All Season")
def compare_teams(team1,team2):
    mt1=iplmatches[((iplmatches['team1']==team1)|(iplmatches['team2']==team1))&((iplmatches['team1']==team2)|(iplmatches['team2']==team2))]
    sns.countplot(x='season', hue='winner',data=mt1,palette=("yellow","blue"))
    sns.color_palette("ch:3.5,-.2,dark=.3")
    plt.xticks(rotation='vertical')
    leg = plt.legend( loc = 'upper center')
    fig=plt.gcf()
    fig.set_size_inches(10,6)
    plt.title("Compare Performance of "+team1 +" and "+team2)
    plt.show()
compare_teams("CSK","MI")
compare_teams("MI","KKR")
compare_teams("CSK","KKR")
finals=matchesdata.drop_duplicates(subset=['season'],keep='last')
finals=finals[['id','season','city','team1','team2','toss_winner','toss_decision','winner']]
most_finals=pd.concat([finals['team1'],finals['team2']]).value_counts().reset_index()
most_finals.rename({'index':'team',0:'count'},axis=1,inplace=True)
xyz=finals['winner'].value_counts().reset_index()
most_finals=most_finals.merge(xyz,left_on='team',right_on='index',how='outer')
most_finals=most_finals.replace(np.NaN,0)
most_finals.drop('index',axis=1,inplace=True)
most_finals.set_index('team',inplace=True)
most_finals.rename({'count':'finals_played','winner':'won_count'},inplace=True,axis=1)
most_finals.plot.bar(width=0.8,color=("orange","b"))
plt.gcf().set_size_inches(10,6)
plt.title("All Finals Played and Won Counts ")
plt.show()