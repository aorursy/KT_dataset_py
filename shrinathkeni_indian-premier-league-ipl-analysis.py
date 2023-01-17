import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snd
import numpy as np
%matplotlib inline
!pip install chart_studio
import chart_studio.plotly as py
from plotly import tools
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=False)
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
data_ipl=pd.read_csv("../input/ipldata/matches.csv")
data_ipl.head(3)
data_ipl=data_ipl.drop(["date","umpire1","umpire2","umpire3","id","dl_applied"],axis=1)
data_ipl["team1"].value_counts()
data_ipl=data_ipl.replace({"Rising Pune Supergiants":"Rising Pune Supergiant"},regex=True)
data_ipl=data_ipl.replace({"Delhi Daredevils":"Delhi Capitals"},regex=True)
data_ipl=data_ipl.replace({"Deccan Chargers":"Sunrisers Hyderabad"},regex=True)
plt.figure(figsize=(10,5))
#plt.plot.bar()
snd.set_style('whitegrid')
snd.set_context('notebook')
snd.set_color_codes("bright")
plt.xticks(rotation=90,fontsize=12)
winnerplot = pd.DataFrame(data_ipl['winner'].value_counts())
snd.barplot(y = winnerplot["winner"] , x = winnerplot.index, data=winnerplot,palette="rocket",color="blue")
#plt.bar(winnerplot.index,winnerplot["winner"])
plt.title('Total wins of each team',fontsize=20)
plt.xlabel('Teams',fontsize=20)
plt.ylabel("Total wins from 2008-2019",fontsize=20)
count=0
for i in winnerplot['winner']:
    plt.text(count-0.20,i+1,str(i),size=15,color='black',rotation=0)
    count+=1
plt.show()    
    
# Creating a new column "Loser" so that we can plot graphs on who lost the maximum matches
data_ipl.loc[data_ipl["team1"]==data_ipl["winner"],"loser"]=data_ipl["team2"]
data_ipl.loc[data_ipl["team2"]==data_ipl["winner"],"loser"]=data_ipl["team1"]
plt.figure(figsize=(10,5))
#plt.plot.bar()
snd.set_style('whitegrid')
snd.set_context('notebook')
snd.set_color_codes("bright")
plt.xticks(rotation=90,fontsize=12)
loserplot = pd.DataFrame(data_ipl['loser'].value_counts())
snd.barplot(y = loserplot["loser"] , x = loserplot.index, data=loserplot,palette="BuGn_r")
#plt.bar(winnerplot.index,winnerplot["winner"])
plt.title('Total loses of each team',fontsize=20)
plt.xlabel('Teams',fontsize=20)
plt.ylabel("Total loses from 2008-2019",fontsize=20)
count=0
for i in loserplot['loser']:
    plt.text(count-0.20,i+1,str(i),size=15,color='black',rotation=0)
    count+=1
Total_Match=data_ipl["team1"].value_counts()+data_ipl["team2"].value_counts()
Total_Match = Total_Match.rename_axis('Teams').to_frame('Matches_played')
Total_Match["Won"]=winnerplot["winner"]
Total_Match["Lost"]=loserplot["loser"]
Total_Match=Total_Match.sort_values(by="Won",ascending=False)
# Calculating win and Lose Percentage
Total_Match["Win Percent"]=(Total_Match["Won"]/Total_Match["Matches_played"])*100
Total_Match["Lose Percent"]=(Total_Match["Lost"]/Total_Match["Matches_played"])*100
Total_Match=Total_Match.round({"Win Percent": 1,"Lose Percent":1})
Total_Match.head(13)
# Setting Bars
bar1=Total_Match["Matches_played"]
bar2=Total_Match["Won"]
bar3=Total_Match["Lost"]

# set width of bar
barWidth = 0.25


#plt.xticks(rotation=90,fontsize=12)
trace1 = go.Bar(x = Total_Match.index,  y = bar1 ,name='Total Matches',width=barWidth)
trace2 = go.Bar(x = Total_Match.index,  y = bar2,name='Matches_won',marker=dict(color='green'),width=barWidth)
trace3 = go.Bar(x = Total_Match.index, y = bar3,name='Matches_lost',marker=dict(color='red'),width=barWidth)


data = [trace1, trace2, trace3]

layout = go.Layout(title='Match Played, Wins And Loses',xaxis=dict(title='Team'), 
                   yaxis=dict(title='Total Matches'),bargap=0.25,bargroupgap=0.25)

fig = go.Figure(data=data, layout=layout)


iplot(fig)
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=Total_Match.index, values=Total_Match["Win Percent"]),1,1)
fig.update_traces(hoverinfo='label', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)),hole=.3)

fig.add_trace(go.Pie(labels=Total_Match.index, values=Total_Match["Lose Percent"]),1,2)
fig.update_traces(hoverinfo='label+value', textinfo='value', textfont_size=20,
 marker=dict(line=dict(color='#000000', width=2)),hole=.4)

# Add annotations in the center of the donut pies.
fig.update_layout(
    title_text="Win and Lose Percent for Each Team",
    annotations=[dict(text='Win Percent', x=0.15, y=0.5, font_size=16, showarrow=False),
                 dict(text='Lose Percent', x=0.86, font_size=16, showarrow=False)])
fig.show()
plt.figure(figsize=(12,5))
City_Data=data_ipl["city"].value_counts()[:9]
City_Data = City_Data.rename_axis('City').to_frame('Matches_played')
snd.barplot(x = City_Data.index , y = City_Data["Matches_played"], data=City_Data,palette="cubehelix")
plt.show()
#City_teams=.loc[City_teams["City"]==data_ipl["winner"],"loser"]=data_ipl["team2"]
City_teams=data_ipl[["city","team1","team2","winner"]]
City_teams

# Creating a new column "Result" based on Home Wins and Home Lose
City_teams.loc[(City_teams['city'] == "Mumbai") & (City_teams['winner'] == "Mumbai Indians"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Mumbai") & ((City_teams["team1"] == "Mumbai Indians") | (City_teams["team2"] == "Mumbai Indians")) & (City_teams['winner'] != "Mumbai Indians"),"Result"]="Lost"
City_teams.loc[(City_teams['city'] == "Kolkata") & (City_teams['winner'] == "Kolkata Knight Riders"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Kolkata") & ((City_teams["team1"] == "Kolkata Knight Riders") | (City_teams["team2"] == "Kolkata Knight Riders")) & (City_teams['winner'] != "Kolkata Knight Riders"),"Result"]="Lost"
City_teams.loc[(City_teams['city'] == "Delhi") & (City_teams['winner'] == "Delhi Capitals"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Delhi") & ((City_teams["team1"] == "Delhi Capitals") | (City_teams["team2"] == "Capitals")) & (City_teams['winner'] != "Delhi Capitals"),"Result"]="Lost"
City_teams.loc[(City_teams['city'] == "Bangalore") & (City_teams['winner'] == "Royal Challengers Bangalore"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Bangalore") & ((City_teams["team1"] == "Royal Challengers Bangalore") | (City_teams["team2"] == "Royal Challengers Bangalore"))& (City_teams['winner'] != "Royal Challengers Bangalore"),"Result"]="Lost"
City_teams.loc[(City_teams['city'] == "Hyderabad") & (City_teams['winner'] == "Sunrisers Hyderabad"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Hyderabad") & ((City_teams["team1"] == "Sunrisers Hyderabad") | (City_teams["team2"] == "Sunrisers Hyderabad "))  & (City_teams['winner'] != "Sunrisers Hyderabad"),"Result"]="Lost"
City_teams.loc[(City_teams['city'] == "Chennai") & (City_teams['winner'] == "Chennai Super Kings"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Chennai") & ((City_teams["team1"] == "Chennai Super Kings") | (City_teams["team2"] == "Chennai Super Kings"))& (City_teams['winner'] != "Chennai Super Kings"),"Result"]="Lost"
City_teams.loc[(City_teams['city'] == "Jaipur") & (City_teams['winner'] == "Rajasthan Royals"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Jaipur")& ((City_teams["team1"] == "Rajasthan Royals") | (City_teams["team2"] == "Rajasthan Royals")) & (City_teams['winner'] != "Rajasthan Royals"),"Result"]="Lost"
City_teams.loc[(City_teams['city'] == "Chandigarh") & (City_teams['winner'] == "Kings XI Punjab"),"Result"]="Won"
City_teams.loc[(City_teams['city'] == "Chandigarh") & ((City_teams["team1"] == "Kings XI Punjab") | (City_teams["team2"] == "Kings XI Punjab")) & (City_teams['winner'] != "Kings XI Punjab"),"Result"]="Lost"

# Dropping NaN values
City_teams=City_teams.dropna()
# Grouping By City and Result and Unstacking

City_teams_group=City_teams.groupby(["city","Result"]).size().unstack(fill_value=0)
City_teams_group["Teams"]=["Royal Challengers Bangalore","Kings XI Punjab","Chennai Super Kings","Delhi Capitals","Sunrisers Hyderabad","Rajashtan Royals","Koklata Knight Riders","Mumbai Indians"]
City_teams_group["Total Played"]=City_teams_group["Won"]+City_teams_group["Lost"]
column_names = ["Teams","Total Played", "Won","Lost"]
City_teams_group=City_teams_group.reindex(columns= column_names)
City_teams_group.head(8)

#plt.figure(figsize=(12,8))
#City_teams_group.plot(figsize=(12,8))
City_teams_group.plot.bar("Teams",width = 0.7,figsize=(12,8),color=["#011627","#f71735","#41ead4"])
plt.show()
City_teams_group_total=City_teams_group[["Total Played"]]
City_teams_group_total["Home Win Percent"]=(City_teams_group["Won"]/City_teams_group_total["Total Played"])*100
City_teams_group_total["Home Lose Percent"]=(City_teams_group["Lost"]/City_teams_group_total["Total Played"])*100
City_teams_group_total=City_teams_group_total.round({"Home Win Percent": 1,"Home Lose Percent":1})
City_teams_group_total["Teams"]=City_teams_group[["Teams"]]
column_names = ["Teams","Total Played", "Home Win Percent","Home Lose Percent"]
City_teams_group_total=City_teams_group_total.reindex(columns= column_names)
City_teams_group_total.head(8)
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=City_teams_group_total["Teams"], values=City_teams_group_total["Home Win Percent"]),1,1)
fig.update_traces(hoverinfo='label', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)),hole=.3)

fig.add_trace(go.Pie(labels=City_teams_group_total["Teams"], values=City_teams_group_total["Home Lose Percent"]),1,2)
fig.update_traces(hoverinfo='label+value', textinfo='value', textfont_size=20,
 marker=dict(line=dict(color='#000000', width=2)),hole=.48)

# Add annotations in the center of the donut pies.
fig.update_layout(
    title_text="Home Win and Lose Percent",
    annotations=[dict(text="Home Win Percent", x=0.12, y=0.5, font_size=15, showarrow=False),
                dict(text='Home Lose Percent', x=0.88, font_size=14, showarrow=False)])
fig.show()
# Setting Bars
bar2=City_teams_group_total["Home Win Percent"]
bar3=City_teams_group_total["Home Lose Percent"]

# set width of bar
barWidth = 0.25


#plt.xticks(rotation=90,fontsize=12)
trace2 = go.Bar(x = City_teams_group_total["Teams"],  y = bar2,name='Home Win Percent',marker=dict(color='green'),width=barWidth)
trace3 = go.Bar(x = City_teams_group_total["Teams"], y = bar3,name='Home Lose Percent',marker=dict(color='red'),width=barWidth)


data = [trace2, trace3]

layout = go.Layout(title='Home Win Percent Vs Home Lose Percent',xaxis=dict(title='Team'), 
                   yaxis=dict(title='Matches'),bargap=0.25,bargroupgap=0.25)

fig = go.Figure(data=data, layout=layout)


iplot(fig)

All_time_man=data_ipl["player_of_match"].value_counts()[:15]
All_time_man.plot.barh(figsize=(10,8),color=['#220901', '#621708', '#941b0c', '#bc3908', '#f6aa1c',"#00a8e8","red","grey"])
plt.gca().invert_yaxis()
Match_man_2017=data_ipl[["player_of_match"]]
Match_man_2017["Season"]=data_ipl["season"]
Match_man_2017=Match_man_2017[Match_man_2017["Season"] == 2017]
#Match_man_2017=Match_man[(Match_man["Season"].isin(['2008','2009',"2010","2011","2012","2013","2014","2015","2016","2018","2"]))]
Match_man_2017=Match_man_2017["player_of_match"].value_counts()[:10]
Match_man_2017 = Match_man_2017.rename_axis('Players').to_frame('2017: No of Man of the match')

Match_man_2018=data_ipl[["player_of_match"]]
Match_man_2018["Season"]=data_ipl["season"]
Match_man_2018=Match_man_2018[Match_man_2018["Season"] == 2018]
Match_man_2018=Match_man_2018["player_of_match"].value_counts()[:10]
Match_man_2018 = Match_man_2018.rename_axis('Players').to_frame('2018: No of Man of the match')


Match_man_2019=data_ipl[["player_of_match"]]
Match_man_2019["Season"]=data_ipl["season"]
Match_man_2019=Match_man_2019[Match_man_2019["Season"] == 2019]
Match_man_2019=Match_man_2019["player_of_match"].value_counts()[:13]
Match_man_2019 = Match_man_2019.rename_axis('Players').to_frame('2019: No of Man of the match')                        
f, axes = plt.subplots(1, 3,figsize=(20,5))
Match_man_2017.plot.bar(color="#ff6600",ax=axes[0])
Match_man_2018.plot.bar(ax=axes[1])
Match_man_2019.plot.bar(color="#669900", ax=axes[2])
plt.show()
data_toss_bat=data_ipl[["toss_winner","toss_decision","winner"]]
data_toss_bat.loc[((data_toss_bat['toss_winner']) == (data_toss_bat["winner"])) &  (data_toss_bat["toss_decision"] == "bat"),"First Batting Result"]="Won"
data_toss_bat.loc[((data_toss_bat['toss_winner']) != (data_toss_bat["winner"])) &  (data_toss_bat["toss_decision"] == "bat"),"First Batting Result"]="Lost"
data_toss_bat=data_toss_bat.dropna()
data_toss_bat.head()
data_toss_bat=data_toss_bat.groupby(["toss_winner","First Batting Result"]).size().unstack(fill_value=0)
column_names = ["Won","Lost"]
data_toss_bat=data_toss_bat.reindex(columns= column_names)
data_toss_bat
data_toss_bat.plot.bar(width = 0.7,figsize=(12,8),color=["#003049","#fcbf49"])
data_toss_field=data_ipl[["toss_winner","toss_decision","winner"]]
data_toss_field.loc[((data_toss_field['toss_winner']) == (data_toss_field["winner"])) &  (data_toss_field["toss_decision"] == "field"),"First Fielding Result"]="Won"
data_toss_field.loc[((data_toss_field['toss_winner']) != (data_toss_field["winner"])) &  (data_toss_field["toss_decision"] == "field"),"First Fielding Result"]="Lost"
data_toss_field=data_toss_field.dropna()

data_toss_field=data_toss_field.groupby(["toss_winner","First Fielding Result"]).size().unstack(fill_value=0)
column_names = ["Won","Lost"]
data_toss_field=data_toss_field.reindex(columns= column_names)
data_toss_field


data_toss_field.plot.bar(width = 0.7,figsize=(12,8),color=["#003049","#fcbf49"])