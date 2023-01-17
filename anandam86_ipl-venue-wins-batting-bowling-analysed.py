# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import pandas.io.formats.style



Deliveries_Data = pd.read_csv('../input/deliveries_expanded.csv')

Matches_Data = pd.read_csv('../input/matches_expanded.csv')

Teams_Zone_Wise = Matches_Data[["team1","Team_Zone_1"]].drop_duplicates()

Teams_Zone_Wise["Team_Zone_1"].value_counts().plot(kind="pie", figsize = (15,8))

plt.xlabel('Distribution of IPL home teams',fontsize = 10)

plt.ylabel('% Contribution of Matches in IPL',fontsize = 10)

plt.title('IPL Teams v/s Home Zone',fontsize = 15)

print(plt.show())
Overall_Matches_State = pd.Series.to_frame(Matches_Data["State"].value_counts())

Overall_Matches_State['%Contribution'] = (Overall_Matches_State['State']/Overall_Matches_State['State'].sum())*100

Overall_Matches_State['%Contribution'].round(1)

Overall_Matches_State_Plot = Overall_Matches_State['%Contribution'].plot(kind="bar",figsize = (15,5),color = 'darkorange',edgecolor='black', hatch = "X")

plt.xlabel('State in India',fontsize = 10)

plt.ylabel('% Contribution of Matches in IPL',fontsize = 10)

plt.title('State v/s Percentage of IPL matches conducted',fontsize = 15)

print(plt.show())
Overall_Matches_Zone = pd.Series.to_frame(Matches_Data["Zone"].value_counts())

Overall_Matches_Zone['%Contribution'] = (Overall_Matches_Zone['Zone']/Overall_Matches_Zone['Zone'].sum())*100

Overall_Matches_Zone_Plot = Overall_Matches_Zone['%Contribution'].plot(kind="bar",figsize = (15,5),color = 'darkorange',edgecolor='black', hatch = "X")

plt.xlabel('Zone',fontsize = 10)

plt.ylabel('% Contribution of Matches in IPL',fontsize = 10)

plt.title('Zone v/s Percentage of IPL matches conducted',fontsize = 15)

print(plt.show())
import matplotlib.pyplot as plt

Matches_understood_by_year =  pd.Series.to_frame(Matches_Data.groupby('season')['Zone'].value_counts())

Plot_Zone = Matches_Data.groupby('season')['Zone'].value_counts().unstack().plot(kind="bar", figsize = (15,7))

print(Plot_Zone.get_legend().set_bbox_to_anchor((0.15,1)))



plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Number of IPL matches conducted',fontsize = 10)

plt.title('IPL matches distribution 2008 to 2019',fontsize = 15)

print(plt.show())
Matches_Data.groupby('season')['Winner_Zone_Type'].value_counts().unstack().plot(kind="bar", stacked = True, figsize = (15,5))

plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Number of IPL matches conducted',fontsize = 10)

plt.title('IPL matches Wins Analysed - Home V/s Away',fontsize = 15)

print(plt.show())
Matches_Data.groupby('city')['Winner_Zone_Type'].value_counts().unstack().dropna().plot(kind="bar", figsize = (15,5))

plt.xlabel('City in India',fontsize = 10)

plt.ylabel('Number of IPL matches conducted',fontsize = 10)

plt.title('IPL matches Wins Analysed - City Wise',fontsize = 15)

print(plt.show())
def Performance_Team_Venue(Team_Name):

    Performance_ByTeam_Analysed =  Matches_Data[(Matches_Data["Home_City_Team1"]==Team_Name) | (Matches_Data["Home_City_team2"]==Team_Name)]

    print(Performance_ByTeam_Analysed["Winner_Zone_Type"].value_counts().plot(kind="pie", figsize = (5,5)))
Performance_Team_Venue("Mumbai")
def Team_Win_By_Zone(Team_Name):

    Wins_Analysed =  Matches_Data[((Matches_Data["team1"]==Team_Name) | (Matches_Data["team2"]==Team_Name)) & (Matches_Data["winner"]==Team_Name)]

    print(Wins_Analysed['Zone'].value_counts().plot(kind="pie", figsize = (10,5)))

    plt.xlabel('Zone',fontsize = 10)

    #plt.ylabel('Number of IPL matches Played',fontsize = 10)

    plt.title(Team_Name,fontsize = 15)

    print(plt.show())

#Wins_Analysis =  Matches_Data[((Matches_Data["team1"]==Team_Name) | (Matches_Data["team2"]==Team_Name)) & (Matches_Data["winner"]==Team_Name)]

#Wins_Analysis.head(2)

Matches_Zone_Plot = Matches_Data.groupby(['winner','Zone'])['season'].count().unstack().plot(kind="bar",figsize = (15,7))



print(Matches_Zone_Plot.get_legend().set_bbox_to_anchor((1,1)))



plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Number of IPL matches Played',fontsize = 10)

plt.title('IPL teams Wins V/s Zone 2008 to 2019',fontsize = 15)

print(plt.show())
Team_Win_By_Zone('Kolkata Knight Riders')
Distinct_Team = Matches_Data["team1"].unique()

Total_Teams = len(Distinct_Team) 



for i in range(Total_Teams):

    Team_Name = Distinct_Team[i]

    Matches_Data_Toss_Comparision = Matches_Data[["city","Zone","toss_winner","toss_decision","Toss_Win_Zone","winner","Winner_Zone"]]

    Matches_Data_Toss_Comparision = Matches_Data_Toss_Comparision[Matches_Data_Toss_Comparision["toss_winner"]==Team_Name]

    Matches_Data_Toss_Comparision["Win_comparison"] = Matches_Data_Toss_Comparision.apply(lambda x: 'Win' if x["toss_winner"] == x["winner"] else 'Lost', axis=1)

    Matches_Data_Toss_Comparision.groupby(['toss_decision','Zone'])['Win_comparison'].value_counts().unstack().plot(kind="bar", figsize = (15,5))

    plt.xlabel('Toss_decision, Zone',fontsize = 10)

    #plt.ylabel('Ticket Count',fontsize = 10)

    plt.title(Team_Name,fontsize = 15)

    print(plt.show())
#Matches_Data_Toss_Comparision = Matches_Data[["city","Zone","toss_winner","toss_decision","Toss_Win_Zone","winner","Winner_Zone"]]

#Matches_Data_Toss_Comparision = Matches_Data_Toss_Comparision[Matches_Data_Toss_Comparision["toss_winner"]=="Mumbai Indians"]

#Matches_Data_Toss_Comparision["Win_comparison"] = Matches_Data_Toss_Comparision.apply(lambda x: 'Win' if x["toss_winner"] == x["winner"] else 'Lost', axis=1)

#Matches_Data_Toss_Comparision.groupby(['toss_decision','Zone'])['Win_comparison'].value_counts().unstack().plot(kind="bar", stacked = True,figsize = (15,5))



#plt.xlabel('Toss Decision',fontsize = 10)

##plt.ylabel('Number of IPL matches Played',fontsize = 10)

#plt.title('IPL Game Winner V/s Toss Result 2008 to 2019',fontsize = 15)

#print(plt.show())
Deliveries_Data.head(2)



Deliveries_Powerplay = Deliveries_Data.loc[Deliveries_Data["over"] < 7]

#Delivery_team = Deliveries_Powerplay[Deliveries_Powerplay["batting_team"] == "Sunrisers Hyderabad"]

Deliveries_team_analysed = Deliveries_Powerplay.groupby(['batting_team','over'])['total_runs'].sum().unstack().plot(kind="bar",figsize = (15,5))

Deliveries_team_analysed.get_legend().set_bbox_to_anchor((0.18,1))

#Deliveries_team_analysed

#.sort_values(by ="total_runs", ascending = False)



#plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Total Number of IPL runs scored',fontsize = 10)

plt.title('Distribution of runs over wise - Power play',fontsize = 15)

print(plt.show())
Deliveries_Powerplay = Deliveries_Data.loc[(Deliveries_Data["over"] > 7) & (Deliveries_Data["over"] < 15)]

#Delivery_team = Deliveries_Powerplay[Deliveries_Powerplay["batting_team"] == "Sunrisers Hyderabad"]

Deliveries_team_analysed = Deliveries_Powerplay.groupby(['batting_team','over'])['total_runs'].sum().unstack().plot(kind="bar",figsize = (15,5))

Deliveries_team_analysed.get_legend().set_bbox_to_anchor((0.18,1))

#plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Total Number of IPL runs scored',fontsize = 10)

plt.title('Distribution of runs over wise - Middle Overs',fontsize = 15)

print(plt.show())
Deliveries_Powerplay = Deliveries_Data.loc[Deliveries_Data["over"] > 15]

#Delivery_team = Deliveries_Powerplay[Deliveries_Powerplay["batting_team"] == "Sunrisers Hyderabad"]

Deliveries_team_analysed = Deliveries_Powerplay.groupby(['batting_team','over'])['total_runs'].sum().unstack().plot(kind="bar",figsize = (15,5))

Deliveries_team_analysed.get_legend().set_bbox_to_anchor((0.18,1))



#plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Total Number of IPL runs scored',fontsize = 10)

plt.title('Distribution of runs over wise - Death Overs',fontsize = 15)

print(plt.show())
#print(Deliveries_Data.groupby(['batting_team','over_type'])['total_runs'].sum().unstack().plot(kind="bar",figsize = (22,5)))

Contribution_data = Deliveries_Data.groupby(['batting_team','over_type'])['total_runs'].sum().unstack()

Contribution_data["%Runs_Powerplay"] = (Contribution_data['Powerplay_Over']/(Contribution_data['Death_Over'] + Contribution_data['Middle_Over'] + Contribution_data['Powerplay_Over']))*100 

Contribution_data["%Runs_MiddleOvers"] = (Contribution_data['Middle_Over']/(Contribution_data['Death_Over'] + Contribution_data['Middle_Over'] + Contribution_data['Powerplay_Over']))*100 

Contribution_data["%Runs_DeathOvers"] = (Contribution_data['Death_Over']/(Contribution_data['Death_Over'] + Contribution_data['Middle_Over'] + Contribution_data['Powerplay_Over']))*100 

Contribution_data

Plot = Contribution_data[['%Runs_Powerplay','%Runs_MiddleOvers','%Runs_DeathOvers']].sort_values(by ="%Runs_DeathOvers", ascending = False).plot(kind="bar",stacked = True, figsize = (15,5))

print(Plot.get_legend().set_bbox_to_anchor((1,1)))

#plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('% IPL runs scored',fontsize = 10)

plt.title('Distribution of runs by over type',fontsize = 15)

print(plt.show())
Deliveries_extras_analysed = Deliveries_Data[["batting_team","bowling_team","over_type","bowler","batsman","wide_runs","bye_runs","legbye_runs","noball_runs","penalty_runs","batsman_runs","extra_runs","total_runs"]]

Deliveries_extras_analysed.head(2)
Extra_Columns = ["batting_team","bowling_team","over_type","bowler","batsman","wide_runs","bye_runs","legbye_runs","noball_runs","penalty_runs","batsman_runs","extra_runs","total_runs"]

Extra_Columns

Deliveries_extras_analysed["Total_Extras"] =  Deliveries_extras_analysed[Extra_Columns].sum(axis=1)

Deliveries_extras_analysed.head(2)



Extras_Over_type = Deliveries_extras_analysed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack()

Extras_Runs = Deliveries_extras_analysed.groupby(['bowling_team','batting_team'])['extra_runs'].sum().unstack().fillna(0)

Extras_Over_type.plot(kind = "bar",figsize=((18,5))).get_legend().set_bbox_to_anchor((1,1))



#plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Total Number of IPL runs scored',fontsize = 10)

plt.title('Distribution of Extra runs',fontsize = 15)

print(plt.show())
Extras_Overall = Deliveries_extras_analysed.groupby('bowling_team')['extra_runs'].sum()

Extras_Overall.sort_values(ascending = False).plot(kind="bar",figsize = (15,5),color = 'darkorange',edgecolor='black', hatch = "X")



#plt.xlabel('IPL Season',fontsize = 10)

plt.ylabel('Total Number runs in extras',fontsize = 10)

plt.title('Distribution of Extra runs vs IPL team',fontsize = 15)

print(plt.show())
def highlight_max(data, color='Khaki'):

    '''

    highlight the maximum in a Series or DataFrame

    '''

    attr = 'background-color: {}'.format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1

        is_max = data == data.max()

        return [attr if v else '' for v in is_max]

    else:  # from .apply(axis=None)

        is_max = data == data.max().max()

        return pd.DataFrame(np.where(is_max, attr, ''),

                            index=data.index, columns=data.columns)
Extras_Overall_Split_Bye = pd.Series.to_frame(Deliveries_extras_analysed.groupby('bowling_team')['bye_runs'].sum().sort_values(ascending = False))

#Extras_Overall_Split_Bye = Extras_Overall_Split_Bye.reset_index()



Extras_Overall_Split_Noball = pd.Series.to_frame(Deliveries_extras_analysed.groupby('bowling_team')['noball_runs'].sum())#.sort_values(ascending = False))

#Extras_Overall_Split_Noball = Extras_Overall_Split_Noball.reset_index()



Extras_Overall_Split_Wides = pd.Series.to_frame(Deliveries_extras_analysed.groupby('bowling_team')['wide_runs'].sum())#.sort_values(ascending = False))

#Extras_Overall_Split_Wides = Extras_Overall_Split_Wides.reset_index()



Extras_Overall_Split_LegBye = pd.Series.to_frame(Deliveries_extras_analysed.groupby('bowling_team')['legbye_runs'].sum())#.sort_values(ascending = False))

#Extras_Overall_Split_LegBye = Extras_Overall_Split_LegBye.reset_index()



Extras_Overall_Split_Penality = pd.Series.to_frame(Deliveries_extras_analysed.groupby('bowling_team')['penalty_runs'].sum())#.sort_values(ascending = False))



Extras_Overall_Split_Noball



Extras_Overall_Split = pd.DataFrame()



Extras_Overall_Split["Byes"] = Extras_Overall_Split_Bye["bye_runs"]

Extras_Overall_Split["Noballs"] = Extras_Overall_Split_Noball["noball_runs"]

Extras_Overall_Split["Wides"] = Extras_Overall_Split_Wides["wide_runs"]

Extras_Overall_Split["LegByes"] = Extras_Overall_Split_LegBye["legbye_runs"]

Extras_Overall_Split["Penality"] = Extras_Overall_Split_Penality["penalty_runs"]

#print(Extras_Overall_Split)

Extras_Overall_Split = Extras_Overall_Split[["Byes","Noballs","Wides","LegByes","Penality"]]



Extra_Columns_Names = ["Byes","Noballs","Wides","LegByes","Penality"]



Extras_Overall_Split["Total_Extras"] =  Extras_Overall_Split[Extra_Columns_Names].sum(axis=1)

#Extras_Overall_Split["AveragePerSeries"] =  Extras_Overall_Split[Extra_Columns_Names].sum(axis=1)

Extras_Overall_Split ["Mean_extras"] = Deliveries_extras_analysed.groupby('bowling_team')['extra_runs'].mean()

Extras_Overall_Split["Mean_extras"] = Extras_Overall_Split["Mean_extras"]*100

#Extras_Overall_Split.dtype#.set_index('bowling_team')

Extras_Overall_Split = Extras_Overall_Split.sort_values(by="Mean_extras", ascending = False)

Extras_Overall_Split.style.apply(highlight_max)



Extras_By_team = Deliveries_extras_analysed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().fillna(0)

Extras_By_team.style.apply(highlight_max)
Total_By_team = Deliveries_extras_analysed.groupby(['batting_team','over_type'])['total_runs'].sum().unstack().fillna(0)

Total_By_team.style.apply(highlight_max)
Total_Runs = Deliveries_extras_analysed.groupby(['bowling_team','batting_team'])['total_runs'].sum().sort_values(ascending = False).unstack().fillna(0)

Total_Runs.style.apply(highlight_max)
Extras_Runs = Deliveries_extras_analysed.groupby(['bowling_team','batting_team'])['extra_runs'].sum().sort_values(ascending = False).unstack().fillna(0)

Extras_Runs.style.apply(highlight_max)
Delivery_Wise_Data = Deliveries_Data.groupby(['batting_team','batsman_runs'])['bowling_team'].count().sort_values(ascending = False).unstack().fillna(0)

Delivery_Wise_Data.style.apply(highlight_max)
All_1_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 1]

All_1_Detailed.head(2)



All_2_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 2]

All_2_Detailed.head(2)



All_3_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 2]

All_3_Detailed.head(2)



All_4_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 4]

All_4_Detailed.head(2)



All_5_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 5]

All_5_Detailed.head(2)



All_6_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 6]

All_6_Detailed.head(2)



All_7_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 7]

#All_7_Detailed.head(2)



All_8_Detailed = Deliveries_Data[Deliveries_Data["batsman_runs"] == 8]

All_1_Detailed.head(2)
#All_1_Detailed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().plot(kind="bar", figsize = ((20,5)))

#All_2_Detailed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().plot(kind="bar", figsize = ((20,5)))

#All_3_Detailed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().plot(kind="bar", figsize = ((20,5)))

#All_4_Detailed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().plot(kind="bar", figsize = ((15,5)))

#All_5_Detailed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().plot(kind="bar", figsize = ((15,5)))

All_6_Detailed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().plot(kind="bar", figsize = ((15,5)))

#All_7_Detailed.groupby(['bowling_team','over_type'])['extra_runs'].sum().unstack().plot(kind="bar", figsize = ((15,5)))



#plt.xlabel('Toss_decision, Zone',fontsize = 10)

plt.ylabel('Number of Instances',fontsize = 10)

plt.title('IPL Team V/S 1''s run',fontsize = 15)

print(plt.show())
Delivery_matrix = Deliveries_Data.groupby(['batsman','over_type'])['total_runs'].sum().unstack()#.sort_values(by ="over_type", ascending = False)#.plot(kind="bar", figsize = ((15,5)))

Death_Batting = pd.Series.to_frame(Delivery_matrix["Death_Over"].sort_values(ascending=False).head(50))

Powerplay_Batting = pd.Series.to_frame(Delivery_matrix["Powerplay_Over"].sort_values(ascending=False).head(50))

Middleover_Batting = pd.Series.to_frame(Delivery_matrix["Middle_Over"].sort_values(ascending=False).head(50))



print(Death_Batting.head(5))

print(Powerplay_Batting.head(5))

print(Middleover_Batting.head(5))

#Batting_Overall_Split = pd.DataFrame()

#Batting_Overall_Split["Death_Over"] = Death_Batting["Death_Over"]

#Batting_Overall_Split["Powerplay_Over"] = Powerplay_Batting["Powerplay_Over"]

#Batting_Overall_Split["Middle_Over"] = Middleover_Batting["Middle_Over"]



#Batting_Overall_Split = Batting_Overall_Split.sort_values(by="Powerplay_Over", ascending = False)



#Batting_Overall_Split.style.apply(highlight_max)

ax = sns.pairplot(Delivery_matrix, diag_kind="kde", diag_kws=dict(shade=True, bw=.05, vertical=False) )
print(sns.violinplot(x='ball',y='total_runs',data=Deliveries_Data,figsize = ((20,10))).set_title('Total Runs by Ball'))
Type_Of_Wicket = Deliveries_Data

Value_Out = Type_Of_Wicket["dismissal_kind"].value_counts()

Value_Out
Top_Bowler = pd.Series.to_frame(Type_Of_Wicket.groupby('bowler')['player_dismissed'].count().fillna(0).sort_values(ascending = False).head(10))

Top_Bowler.style.apply(highlight_max)
Top_Batsman = pd.Series.to_frame(Deliveries_Data.groupby('batsman')['total_runs'].sum().fillna(0).sort_values(ascending = False).head(10))

Top_Batsman.style.apply(highlight_max)
Type_Of_Wicket_Death = Type_Of_Wicket[Type_Of_Wicket["over_type"] == 'Death_Over']

Type_Of_Wicket_Powerplay = Type_Of_Wicket[Type_Of_Wicket["over_type"] == 'Powerplay_Over']

Type_Of_Wicket_Middle = Type_Of_Wicket[Type_Of_Wicket["over_type"] == 'Middle_Over']



Death_Bowler_Analysis= Type_Of_Wicket_Death.groupby(['bowler','over_type'])['player_dismissed'].count().unstack().fillna(0).sort_values(by="Death_Over", ascending = False).head(10)

Powerplay_Bowler_Analysis= Type_Of_Wicket_Powerplay.groupby(['bowler','over_type'])['player_dismissed'].count().unstack().fillna(0).sort_values(by="Powerplay_Over", ascending = False).head(10)

Middle_Bowler_Analysis= Type_Of_Wicket_Middle.groupby(['bowler','over_type'])['player_dismissed'].count().unstack().fillna(0).sort_values(by="Middle_Over", ascending = False).head(10)





Death_Bowler_Analysis.style.apply(highlight_max)

Powerplay_Bowler_Analysis.style.apply(highlight_max)

Middle_Bowler_Analysis.style.apply(highlight_max)
#Type_Of_Wicket['bowler_Frequency'] = Type_Of_Wicket.groupby('bowler')['bowler'].transform('count')

Type_Of_Wicket_Clean = Type_Of_Wicket.dropna()

Type_Of_Wicket_Clean.head(20)

Type_Of_Wicket_Clean["bowler"].value_counts().sort_values(ascending = False)

Type_Of_Wicket_Clean.groupby(['bowler','over_type'])['player_dismissed'].count().unstack().sort_values(by="Death_Over", ascending = False).head(10).plot(kind="bar", figsize = ((15,5)))

#Type_Of_Wicket_Clean.groupby(['bowler','over_type'])['player_dismissed'].count().unstack().sort_values(by="Middle_Over", ascending = False).head(10).plot(kind="bar", figsize = ((15,5)))

#Type_Of_Wicket_Clean.groupby(['bowler','over_type'])['player_dismissed'].count().unstack().sort_values(by="Powerplay_Over", ascending = False).head(10).plot(kind="bar", figsize = ((15,5)))

plt.xlabel('Bowler',fontsize = 10)

plt.ylabel('Total Wickets Taken',fontsize = 10)

plt.title('Best Death Over bowler - Wickets Wise',fontsize = 15)

print(plt.show())
Wickets = Type_Of_Wicket.groupby(['bowling_team','dismissal_kind'])['dismissal_kind'].count().sort_values(ascending = False).unstack().fillna(0)

Wickets.style.apply(highlight_max)