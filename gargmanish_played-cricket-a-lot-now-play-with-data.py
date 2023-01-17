import numpy as np # for Linear algebra

import pandas as pd # for data manipulation/CSV I/O

import matplotlib.pyplot as plt # for plotting Graphs

import seaborn as sns # for interactive graphs

% pylab inline
deliveries=pd.read_csv("../input/deliveries.csv")

matches =pd.read_csv("../input/matches.csv")
matches.head(2) # the data related to matches
deliveries.head(2) # ball by ball data
#lets Define a function to make graph

# I am usinng this function direct from a notebook "Indian Exploration League by SRK"

def autolabel(rects):

    for rect in rects:

        height = rect.get_height() # here height means the hieght of a rectangle 

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom') # this is used for text in the bars
Batsman_score=deliveries.groupby('batsman')["batsman_runs"].agg(sum).reset_index().sort_values(by="batsman_runs",ascending=False).reset_index(drop=True)

#Lets have a look what is happening 

# we group our deliveries data by Batsman ans then we sum the batsman_runs mean run scored by batsman

# Then sort those values in decreasing order

# reset_index means we are indexing again our data which changed due to first group by then by sort

Top_batsman_score= Batsman_score.iloc[:15,:] # here I am taking only top 10 scorer in IPL

Top_batsman_score # this will show our data
labels = np.array(Top_batsman_score['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.7 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Top_batsman_score['batsman_runs']), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("Top Scorer in IPL")

autolabel(rects)
Batsman_Ball_faced=deliveries.groupby(['batsman'])["ball"].count().reset_index().sort_values(by="ball",ascending=False).reset_index(drop=True)

Batsman_Ball_faced_Top=Batsman_Ball_faced.iloc[:15,:] # batsman with most ball faced
labels = np.array(Batsman_Ball_faced_Top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.7 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Batsman_Ball_faced_Top['ball']), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("Ball faced by Batsman in IPL")

autolabel(rects)
Batsman_strike_rate= pd.merge(Batsman_score,Batsman_Ball_faced,on="batsman",how='outer')

# here I am merging two data frames batsman run and batsman ball to get strike rate

Batsman_strike_rate= Batsman_strike_rate[Batsman_strike_rate["batsman_runs"]>=500]

# here I am taking only batsmen habing more than 500 runs under their belts

Batsman_strike_rate["strike_rate"]= (Batsman_strike_rate["batsman_runs"]/Batsman_strike_rate["ball"])*100

# formula for strike rate

Batsman_strike_rate= Batsman_strike_rate[["batsman","strike_rate"]]

# removing other coloumns from here only keeping batsman and strike rate

Batsman_strike_rate=Batsman_strike_rate.sort_values(by="strike_rate",ascending=False).reset_index(drop=True)

Batsman_strike_rate.iloc[:20,:]

# here you can have a look at the table
Batsman_strike_rate_Top=Batsman_strike_rate.iloc[:15,:]

labels = np.array(Batsman_strike_rate_Top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Batsman_strike_rate_Top['strike_rate']), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Strike Rate")

ax.set_title("Most Destructive Player in IPL")

autolabel(rects)
Batsman_dotballs=deliveries[deliveries["extra_runs"]==0].groupby(['batsman'])["batsman_runs"].agg(lambda x: (x==0).sum()).reset_index().sort_values(by="batsman_runs",ascending=False).reset_index(drop=True)

# here the dot ball for a batsman will be if it is a legal delivery and batsman didn't score a run on it

Batsman_dotballs.columns = ["batsman","No_of_Balls"]

Batsman_dotballs.iloc[:20,:]
Batsman_dotballs_Top = Batsman_dotballs.iloc[:15,:]

labels = np.array(Batsman_dotballs_Top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.6 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Batsman_dotballs_Top["No_of_Balls"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No. of Dot Balls")

autolabel(rects)

Percentage_of_dot_balls=pd.merge(Batsman_Ball_faced,Batsman_dotballs,on="batsman",how='outer')

Percentage_of_dot_balls["% of dot balls"] = (Percentage_of_dot_balls["No_of_Balls"]/Percentage_of_dot_balls["ball"])*100

Percentage_of_dot_balls=Percentage_of_dot_balls[Percentage_of_dot_balls["ball"]>300].reset_index(drop=True)

Percentage_of_dot_balls_top=Percentage_of_dot_balls.sort_values(by="% of dot balls",ascending=False).reset_index(drop=True).iloc[:15,:]

Percentage_of_dot_balls_top.iloc[:20,:]
labels = np.array(Percentage_of_dot_balls_top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Percentage_of_dot_balls_top['% of dot balls']), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Percentage")

ax.set_title("% of Dot Balls(No. of ball faced >300)")

autolabel(rects)
Batsman_fours=deliveries.groupby(['batsman'])["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index().sort_values(by="batsman_runs",ascending=False).reset_index(drop=True)

# taking only batsman runs where runs = 4 

Batsman_fours.columns = ["batsman", "No. of 4s"]

Batsman_fours.iloc[:20,:]
Batsman_fours_Top=Batsman_fours.iloc[:15,:]# top 10 player with most number of 4s

labels = np.array(Batsman_fours_Top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Batsman_fours_Top['No. of 4s']), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No_of_4s")

autolabel(rects)

Batsman_sixes=deliveries.groupby("batsman")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index().sort_values(by="batsman_runs",ascending=False).reset_index(drop=True)

Batsman_sixes.columns= ["batsman","No_of_6s"]

Batsman_sixes.iloc[:20,:]
Batsman_sixes_Top=Batsman_sixes.iloc[:15,:]

labels = np.array(Batsman_sixes_Top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Batsman_sixes_Top['No_of_6s']), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No_of_6s")

autolabel(rects)



Batsman_dotballs=deliveries[deliveries["extra_runs"]==0].groupby(['batsman'])["batsman_runs"].agg(lambda x: (x==0).sum()).reset_index().sort_values(by="batsman_runs",ascending=False).reset_index(drop=True)

# here the dot ball for a batsman will be if it is a legal delivery and batsman didn't score a run on it

Batsman_dotballs
Batsman_Score_by_Match= deliveries[["match_id","batsman","batsman_runs"]]

Batsman_Score_by_Match=Batsman_Score_by_Match.groupby(["match_id","batsman"]).sum()

Batsman_with_centuries=Batsman_Score_by_Match[Batsman_Score_by_Match["batsman_runs"]>=100].reset_index()

Batsman_with_centuries=Batsman_with_centuries.groupby("batsman")["match_id"].count().reset_index().sort_values(by="match_id",ascending=False).reset_index(drop=True)

Batsman_with_centuries.columns=["batsman","No_of_centuries"]

Batsman_with_centuries
Batsman_with_Half_centuries=Batsman_Score_by_Match[Batsman_Score_by_Match["batsman_runs"]>=50].reset_index()

Batsman_with_Half_centuries=Batsman_with_Half_centuries.groupby("batsman")["match_id"].count().reset_index().sort_values(by="match_id",ascending=False).reset_index(drop=True)

Batsman_with_Half_centuries_top=Batsman_with_Half_centuries[Batsman_with_Half_centuries["match_id"]>=10]

Batsman_with_Half_centuries_top.columns = ["batsman","No_of_half_centuries"]

Batsman_with_Half_centuries_top
Striker_on_last_ball=deliveries[["match_id","batsman"]][(deliveries["over"]==20) & (deliveries["ball"]==6)]

# this will be last ball of a match

Striker_on_last_ball=Striker_on_last_ball.groupby("batsman")["match_id"].count().reset_index().sort_values(by="match_id",ascending=False).reset_index(drop=True)

# this will count the number of instances for the batsman when he was on strike for last ball

Striker_on_last_ball.columns=["batsman","No_of_Matches"]



Non_Striker_on_last_ball=deliveries[["match_id","non_striker"]][(deliveries["over"]==20) & (deliveries["ball"]==6)]

Non_Striker_on_last_ball=Non_Striker_on_last_ball.groupby("non_striker")["match_id"].count().reset_index().sort_values(by="match_id",ascending=False).reset_index(drop=True)

# same for Non_striker

Non_Striker_on_last_ball.columns=["batsman","No_of_Matches"]

Players_on_last_ball= pd.concat([Striker_on_last_ball,Non_Striker_on_last_ball],ignore_index=True)

# these are the total players concated

Players_on_last_ball=Players_on_last_ball.groupby("batsman")["No_of_Matches"].sum().reset_index().sort_values("No_of_Matches",ascending=False).reset_index(drop=True)

Players_on_last_ball.head(2)

# this give the number of instances when a player was at either striker or non striker end on last ball
player_dismissed_on_last_ball=deliveries[["match_id","player_dismissed"]][(deliveries["over"]==20) & (deliveries["ball"]==6)]

player_dismissed_on_last_ball=player_dismissed_on_last_ball.groupby("player_dismissed")["match_id"].count().reset_index().sort_values(by="match_id",ascending=False).reset_index(drop=True)

player_dismissed_on_last_ball.columns=["batsman","No_of_Matches"]

player_dismissed_on_last_ball.head(2)

# this gives the no of instances when a player was dismissed on last ball

Batsman_Not_out=pd.merge(Players_on_last_ball,player_dismissed_on_last_ball,on ="batsman",how="outer")

Batsman_Not_out=Batsman_Not_out.fillna(0)

Batsman_Not_out["Not_out"]=Batsman_Not_out["No_of_Matches_x"]-Batsman_Not_out["No_of_Matches_y"]

Batsman_Not_out.drop("No_of_Matches_x",axis=1,inplace=True)



Batsman_Not_out.drop("No_of_Matches_y",axis=1,inplace=True)

Batsman_Not_out=Batsman_Not_out.sort_values(by="Not_out",ascending=False)

Batsman_Not_out
Batsman_Not_out_Top =Batsman_Not_out.iloc[:15,:]

labels = np.array(Batsman_Not_out_Top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Batsman_Not_out_Top['Not_out']), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No_of_Not_out")

autolabel(rects)





No_Matches_player_dismissed = deliveries[["match_id","player_dismissed"]]

No_Matches_player_dismissed =No_Matches_player_dismissed .groupby("player_dismissed")["match_id"].count().reset_index().sort_values(by="match_id",ascending=False).reset_index(drop=True)

No_Matches_player_dismissed.columns=["batsman","No_of Matches"]

No_Matches_player_dismissed .head(2)

Batsman_Average=pd.merge(Batsman_score,No_Matches_player_dismissed ,on="batsman")

#merging the score and match played by batsman

Batsman_Average=Batsman_Average[Batsman_Average["batsman_runs"]>=500]

# taking Average for those player for having more than 500 runs under thier belt

Batsman_Average["Average"]=(Batsman_Average["batsman_runs"]/Batsman_Average["No_of Matches"])

Batsman_Average=Batsman_Average.sort_values(by="Average",ascending=False).reset_index(drop=True)



Batsman_Average.iloc[:20,:]
Batsman_Average_Top=Batsman_Average.iloc[:15,:]

labels = np.array(Batsman_Average_Top['batsman'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Batsman_Average_Top["Average"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Average")

ax.set_title("Average of Batsmen(runs>500)")

autolabel(rects)







Bowler_wicket = deliveries[deliveries["dismissal_kind"] != "run out"]

Bowler_wicket= Bowler_wicket[["bowler","player_dismissed"]]

Bowler_wicket = Bowler_wicket.dropna()

Bowler_wicket= Bowler_wicket.groupby("bowler")["player_dismissed"].count().reset_index().sort_values(by="player_dismissed",ascending=False).reset_index(drop=True)

Bowler_wicket.columns=["bowler","Wickets"]

Bowler_wicket.iloc[:20,:]
Bowler_wicket_Top =Bowler_wicket.iloc[:15,:]

labels = np.array(Bowler_wicket_Top['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_wicket_Top["Wickets"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title(" No of Wickets")

autolabel(rects)

Bowler_over= deliveries[deliveries["extra_runs"]==0][["ball","bowler"]] # here extra balls I am not considering include in over

Bowler_over= Bowler_over.groupby("bowler")["ball"].count().reset_index().sort_values(by="ball",ascending=False).reset_index(drop=True)

Bowler_over["No_of_Overs"]=(Bowler_over["ball"]/6)

Bowler_over.iloc[:20,:]
Bowler_over_Top =Bowler_over.iloc[:15,:]

labels = np.array(Bowler_over_Top['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_over_Top["No_of_Overs"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No. of overs")

autolabel(rects)



Bowler_dotball= deliveries.groupby("bowler")["total_runs"].agg(lambda x: (x==0).sum()).reset_index().sort_values(by="total_runs",ascending=False).reset_index(drop=True)

Bowler_dotball.columns=["bowler","No_of_balls"]

Bowler_dotball_Top = Bowler_dotball.iloc[:15,:]

labels = np.array(Bowler_dotball_Top['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_dotball_Top["No_of_balls"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No. of dot balls")

autolabel(rects)

Bowler_extraball= deliveries.groupby("bowler")["extra_runs"].agg(lambda x: (x>0).sum()).reset_index().sort_values(by="extra_runs",ascending=False).reset_index(drop=True)

Bowler_extraball.columns=["bowler","No_of_balls"]

Bowler_extraball_Top = Bowler_extraball.iloc[:15,:]

labels = np.array(Bowler_extraball_Top ['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_extraball_Top ["No_of_balls"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No. of extra balls")

autolabel(rects)

Bowler_Runs= deliveries.groupby("bowler")["total_runs"].sum().reset_index().sort_values(by="total_runs",ascending=False).reset_index(drop=True)

Bowler_Runs.columns=["bowler","Total_runs_given"]



Bowler_Runs_Top = Bowler_Runs.iloc[:10,:]

labels = np.array(Bowler_Runs_Top['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.7 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_Runs_Top["Total_runs_given"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No of Runs given by bowlers")

autolabel(rects)
Bowler_economy= pd.merge(Bowler_Runs,Bowler_over,on="bowler")
Bowler_economy=Bowler_economy[Bowler_economy["No_of_Overs"]>50] # taking only bowler with minimum 50 ov

Bowler_economy["Economy_rate"]=Bowler_economy["Total_runs_given"]/Bowler_economy["No_of_Overs"]

Bowler_economy=Bowler_economy.sort_values(by="Economy_rate").reset_index(drop=True)

Bowler_economy.iloc[:20,:]
Bowler_economy_Top = Bowler_economy.iloc[:15,:]

labels = np.array(Bowler_economy_Top['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_economy_Top["Economy_rate"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Eeonomy")

ax.set_title("Economy_rate(ovesr>50)")

autolabel(rects)

Bowler_Strike_rate = pd.merge(Bowler_over,Bowler_wicket,on="bowler")

Bowler_Strike_rate= Bowler_Strike_rate[Bowler_Strike_rate["No_of_Overs"]>50] # taking only bowler with minimum 50 overs

Bowler_Strike_rate["Strike_rate"]=Bowler_Strike_rate["ball"]/Bowler_Strike_rate["Wickets"]

Bowler_Strike_rate=Bowler_Strike_rate.sort_values(by="Strike_rate").reset_index(drop=True)

Bowler_Strike_rate.iloc[:20,:]
Bowler_Strike_rate_Top = Bowler_Strike_rate.iloc[:15,:]

labels = np.array(Bowler_Strike_rate_Top['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_Strike_rate_Top ["Strike_rate"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("strike ")

ax.set_title("Strike_rate(ovesr>50)")

autolabel(rects)

Bowler_Average = pd.merge(Bowler_Runs,Bowler_wicket,on="bowler")

Bowler_Average= Bowler_Average[Bowler_Average["Wickets"]>20] # taking only bowler with minimum 20 wickets

Bowler_Average["Average"]=Bowler_Average["Total_runs_given"]/Bowler_Average["Wickets"]

Bowler_Average=Bowler_Average.sort_values(by="Average").reset_index(drop=True)

Bowler_Average.iloc[:20,:]
Bowler_Average_Top = Bowler_Average.iloc[:15,:]

labels = np.array(Bowler_Average_Top['bowler'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Bowler_Average_Top["Average"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel(" Average")

ax.set_title("Average(Wickets>20)")

autolabel(rects)

Bowler_Strike_rate.drop("ball",axis=1,inplace=True)

Bowler_Average.drop("Wickets",axis=1,inplace=True) # removing Unnesecary data

Bowler_economy1=Bowler_economy[["bowler","Economy_rate"]]
Bowler_Report = pd.merge(Bowler_Strike_rate,Bowler_Average,on="bowler")

Bowler_Report1=pd.merge(Bowler_Report,Bowler_economy1,on="bowler")

Bowler_Report= Bowler_Report1[["bowler","No_of_Overs","Wickets","Total_runs_given","Economy_rate","Strike_rate","Average"]]
Bowler_Report = Bowler_Report.sort_values(by="Wickets",ascending=False).reset_index(drop=True)

Bowler_Report

                 
Wicket_Keepers = deliveries[deliveries["dismissal_kind"]=="stumped"][["fielder"]]
Wicket_Keepers= Wicket_Keepers.drop_duplicates().reset_index(drop=True)
Wicket_Keepers # in this list some part time wicket keeper also there
# lets First calculate the no of stumps by a wicket keeper

Wicket_keeper_data = pd.merge(deliveries,Wicket_Keepers,on="fielder") 
Wicket_keeper_Stumps =Wicket_keeper_data.groupby("fielder")["dismissal_kind"].agg(lambda x : (x=="stumped").sum()).reset_index().sort_values(by="dismissal_kind",ascending=False).reset_index(drop=True)

Wicket_keeper_Stumps.columns= ["Wicket_keeper", "Stumps"]

Wicket_keeper_Stumps
# I have Checked the stumps data with records they are matching
Wicket_keeper_Catches =Wicket_keeper_data.groupby("fielder")["dismissal_kind"].agg(lambda x : (x=="caught").sum()).reset_index().sort_values(by="dismissal_kind",ascending=False).reset_index(drop=True)

Wicket_keeper_Catches.columns =["Wicket_Keeper","No_of_cathes"]

Wicket_keeper_Catches
# these Catches data are catch taken by player either as a fielder or wicket keeper
Fielder_data = deliveries[["dismissal_kind","fielder"]].dropna() # bowled data will also be droped because  no fielder involve

Fielder_Runout=Fielder_data.groupby("fielder")["dismissal_kind"].agg(lambda x: (x=="run out").sum()).reset_index().sort_values(by="dismissal_kind",ascending=False).reset_index(drop=True)

Fielder_Runout_Top = Fielder_Runout.iloc[:15,:]

Fielder_Runout_Top.columns=["fielder","No_of_Rumouts"]

labels = np.array(Fielder_Runout_Top['fielder'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(Fielder_Runout_Top["No_of_Rumouts"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No_of_Rumouts")

autolabel(rects)
fielder_catch=deliveries[["bowler","dismissal_kind","fielder"]]

fielder_catch.fielder.fillna(fielder_catch.bowler,inplace=True) # this might be silly

# but the catches taken by biowler when he was bowling i.e. dismissal kind is Caught and bowled

#that also will counted as no of catches by that bowler

# but now fielder is not avaliable for this

#so i replace all by bowler
fielder_catch= fielder_catch.groupby("fielder")["dismissal_kind"].agg(lambda x : ((x=="caught") | (x=="caught and bowled")).sum()).reset_index()
fielder_catch= fielder_catch.sort_values(by="dismissal_kind",ascending=False).reset_index(drop=True)
fielder_catch.columns=["fielder","No.of Catches"]

fielder_catch_top=fielder_catch.iloc[:15,:]

labels = np.array(fielder_catch_top['fielder'])# x axis label of graph

ind = np.arange(len(labels)) # making them as indexes

width = 0.5 # width of rectangle

fig, ax = plt.subplots() # for figure

rects = ax.bar(ind, np.array(fielder_catch_top["No.of Catches"]), width=width, color='blue')# here ind is X

#and np.array(Batsman_Ball_faced_Top['ball']) value is height

ax.set_xticks(ind+((width)/2.))# this is to define the postion in x axis 

ax.set_xticklabels(labels, rotation='vertical') # this is for label x axis

ax.set_ylabel("Count")

ax.set_title("No_of_Catches")

autolabel(rects)
# Suresh raina is top in the list
#now we will start a little analysis of every season
# I want to know the Purple and orange cap season by season
Season_data = matches[["id","season"]]

Season_data.columns = ["match_id","season"]

Season_data=pd.merge(deliveries,Season_data,on="match_id")

Season_orange_cap = Season_data.groupby(["season","batsman"])["batsman_runs"].sum().reset_index().sort_values(by="batsman_runs",ascending=False).reset_index(drop=True)

Season_orange_cap= Season_orange_cap.drop_duplicates(subset=["season"],keep="first").sort_values(by="season").reset_index(drop=True)

Season_orange_cap
Season_purple_cap=Season_data[Season_data["dismissal_kind"]!="run out"]

Season_purple_cap=Season_data.groupby(["season","bowler"])["dismissal_kind"].count().reset_index().sort_values(by="dismissal_kind",ascending=False).reset_index(drop=True)

Season_purple_cap= Season_purple_cap.drop_duplicates(subset=["season"],keep="first").sort_values(by="season").reset_index(drop=True)

Season_purple_cap.columns= ["Season","Bowler","Wicket_taken"]

Season_purple_cap
# IPL Winner Team

IPL_Winner=matches.drop_duplicates(subset=["season"],keep="last").sort_values(by="season").reset_index(drop=True)

IPL_Winner=IPL_Winner[["season","winner"]]

IPL_Winner
Season_Highest=Season_data.groupby(["match_id","season","inning","batting_team","bowling_team"])["total_runs"].sum().reset_index().sort_values(by="total_runs",ascending=False)
Season_Highest=Season_Highest.drop_duplicates(subset=["season"],keep="first").sort_values(by="season").reset_index(drop=True)

#Season_Highest.drop(["match_id","inning"],axis=1,inplace=True)

Season_Highest
# no o sixes by season
Season_sixes =Season_data.groupby("season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()

Season_sixes
Season_fours =Season_data.groupby("season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()

Season_fours
Season_total_runs =Season_data.groupby("season")["total_runs"].sum().reset_index()

Season_total_runs
Season_Wickets =Season_data.groupby(["season","dismissal_kind"])["player_dismissed"].count()

Season_Wickets
# Season Wise no of matches
sns.countplot(x="season",data=matches)
# Season Wise Toss Decision
sns.countplot(x="season",hue="toss_decision",data=matches)