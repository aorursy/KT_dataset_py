from IPython.display import HTML
style = """
<style>
    .header1 { font-family:'Arial';font-size:30px; color:Black; font-weight:800;}
    .header2 { 
        font-family:'Arial';
        font-size:18px; 
        color:Black; 
        font-weight:600;
        border-bottom: 1px solid; 
        margin-bottom: 8px;
        margin-top: 8px;
        width: 100%;
        
    }
    .header3 { font-family:'Arial';font-size:16px; color:Black; font-weight:600;}
    .para { font-family:'Arial';font-size:14px; color:Black;}
    .flex-columns {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }
    .flex-container {
         padding: 20px;
    }
    
    .flex-container-large {
         padding: 20px;
         max-width: 40%;
    }
    
    .flex-container-small {
         padding: 20px;
         max-width: 17.5%;
    }
    
    .list-items {
        margin: 10px;
    }
    
    .list-items li {
        color: #3692CC;
        font-weight: 500;
    }
</style>
"""
HTML(style)
import pandas as pd
import glob
from plotly import offline
import plotly.graph_objs as go
import os
import numpy as np
pd.set_option('max.columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
offline.init_notebook_mode()
config = dict(showLink=False)
from IPython.display import IFrame
from IPython.display import display
from IPython.display import Image

#Load provided files
play_information = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
video_footage_injury = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
video_footage_control = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-control.csv')
game_data = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')
player_jersey = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
player_role = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
# replace ". " with "," to make the descriptions easier to separate
play_information.PlayDescription=play_information.PlayDescription.str.replace('. ',',',regex=False)

#Look for key words.  0 = not found, 1 = found
play_information['Center']=np.where(play_information.PlayDescription.str.find('Center')==-1,0,1)
play_information['Penalty']=np.where(play_information.PlayDescription.str.find('PENALTY')==-1,0,1)
play_information['Downed']=np.where(play_information.PlayDescription.str.find('downed')==-1,0,1)
play_information['Touchback']=np.where(play_information.PlayDescription.str.find('Touchback')==-1,0,1)
play_information['OutofBounds']=np.where(play_information.PlayDescription.str.find('out of bounds')==-1,0,1)
play_information['FairCatch']=np.where(play_information.PlayDescription.str.find('fair catch')==-1,0,1)
play_information['TD']=np.where(play_information.PlayDescription.str.find('TOUCHDOWN')==-1,0,1)
play_information['Injury']=np.where(play_information.PlayDescription.str.find('njur')==-1,0,1)

#Divide the play description into 4 parts -- Punter, Center, Returner, Extra 
play_information[['Punter','Center','Returner','Extra']]=play_information.PlayDescription.str.split(',', expand=True, n=3)
play_information.PlayDescription=play_information.PlayDescription.str.replace('(Punt formation)','',regex=False)
play_information['ReturnYards'] = play_information.Returner.str.extract(r'for.([\w]+)')
play_information['PuntYards'] = play_information.Punter.str.extract(r'punts.([\w]+)')

#Clean up errors
play_information['PuntYards'] = pd.to_numeric(play_information['PuntYards'], errors='coerce')
play_information['ReturnYards'] = pd.to_numeric(play_information['ReturnYards'].str.replace('no','0',regex=False), errors='coerce')
play_information['PuntYards_qcut'] = pd.qcut(play_information['PuntYards'],5)
play_information['ReturnYards_qcut'] = pd.qcut(play_information['ReturnYards'],5)

#Calculate how close the punting team was to a TD when they punted.
#If the ball is on the Possession team's side, this will calculate the distance to the endzone
#If the ball is on the receiver team's side, this will return 1.
play_information['YardsforTD_possession']=100-pd.to_numeric(play_information.apply(lambda x : x['YardLine'].replace((x['Poss_Team']+' '),''),1), errors='coerce')
play_information['YardsforTD_receive']=play_information.YardLine.str.extract(r'([\d]+)')
play_information.YardsforTD_receive = pd.to_numeric(play_information.YardsforTD_receive)
#Find the larger of the two to get the real value.  Probably could have done this in one line, but 
#I think it's clearer this way
play_information['YardsforTD']= play_information[['YardsforTD_possession','YardsforTD_receive']].max(axis=1)
play_information['YardsforTD_qcut'] = pd.qcut(play_information['YardsforTD'],5)
#Calculate netyards by starting with punt yards and subtracting either return or touchback
play_information['NetYards'] = play_information.PuntYards
play_information['NetYards'] = play_information['NetYards'] - play_information.Touchback.apply(lambda x: 20 if x > 0 else 0)
play_information['NetYards'] =  play_information['NetYards'] - play_information.ReturnYards.apply(lambda x: 0 if pd.isna(x) else x)
play_information['NetYards_qcut'] = pd.qcut(play_information['NetYards'],5)
#Make the score usable
play_information[['Score_Home','Score_Visiting']] = play_information.Score_Home_Visiting.str.split('-',expand=True,n=2)
play_information['Score_Diff'] = abs(pd.to_numeric(play_information.Score_Home) - pd.to_numeric(play_information.Score_Visiting))
play_information['Score_Diff_qcut'] = pd.qcut(play_information['Score_Diff'],4)

#Determine receiving team
play_information['Rec_Team'] =play_information.apply(lambda x : x['Home_Team_Visit_Team'].replace(x['Poss_Team'],''),1)
play_information['Rec_Team'] = play_information.apply(lambda x : x['Rec_Team'].replace('-',''),1)

#Delete intermediate variables
play_information.drop(['Punter','Center','Returner','Extra','YardsforTD_possession','YardsforTD_receive'], axis=1, inplace=True)

#play_information.head()
#Injury Data --  Need only URL column.  All other columns are in play_information
video_footage_injury = video_footage_injury[['season','gamekey','playid','PREVIEW LINK (5000K)']]
video_footage_injury.columns = ['Season','GameKey','PlayID','URL']

#Control data -- Need only URL column.  All other columns are in play_information
video_footage_control = video_footage_control[['season','gamekey','playid','Preview Link']]
video_footage_control.columns = ['Season','GameKey','PlayID','URL']

#for injury data, add in video review columns
video = pd.merge(video_footage_injury,video_review, on=['GameKey','PlayID'])
#video.head()
Image("../input/images/Players_Roles.png")
#File I created that adds team (coverage vs. return), side (left, right, center)
# and Group (Gunner, Jammer, Dline, Wing)
expanded_role = pd.read_csv('../input/extradata/Expanded_Roles.csv')

#Players rarely change positions but they frequently change numbers (pre-season vs regular season, 2016 vs 2017)
player_jersey = player_jersey.groupby('GSISID').agg({'Number': ', '.join, 
                             'Position': 'first' }).reset_index()

#Merge 3 DFs into one DF
player = pd.merge(player_role,player_jersey,on='GSISID')
player = pd.merge(player,expanded_role,on="Role")

#Formation - How players are lined up
RoleGroup = player.groupby(['Season_Year','GameKey','PlayID','Cov_Ret','Group']).agg({'Role': 'count'}).reset_index()
Role = RoleGroup.sort_values(['Season_Year','GameKey','PlayID','Role'])
Role['RoleCount'] = Role.Group + '-' + Role.Role.map(str)
Role =  Role.groupby(['Season_Year','GameKey','PlayID','Cov_Ret']).agg({'RoleCount': ', '.join}).reset_index()
Role.columns = ['Season_Year','GameKey','PlayID','Cov_Ret','Formation']
#Add Concussion column to Video to help separate concussion plays
video['Concussion']='YES'
Role_Con = pd.merge(Role,video,on=['Season_Year','GameKey','PlayID'],how='left')
Role_Con.Concussion = Role_Con.Concussion.fillna('NO')

#Group by formation.  Separate formations by coverage and return teams
Role_Agg = Role_Con.groupby(['Cov_Ret','Formation','Concussion'])['Season_Year'].agg('count').reset_index()

#Separate Yes and No's and join them to compare them.
#Many formations had no concussions.  We may revisit them later.
Role_Yes = Role_Agg[Role_Agg.Concussion=='YES']
Role_No = Role_Agg[Role_Agg.Concussion=='NO']

Role_Percent=pd.merge(Role_Yes,Role_No,on=['Cov_Ret','Formation'],suffixes=['_Yes','_No'],how='left')

#The second calculation is a little non-intuitive.  I could have hard coded, but I 
# wanted the code to work if the data changed
# Len(Role_Con) = 2*plays because each play has a coverage formation and return formation
Role_Percent['Season_Year_Yes_Percent'] = Role_Percent.Season_Year_Yes/len(video)
Role_Percent.Season_Year_No = Role_Percent.Season_Year_No/(len(Role_Con)/2-len(video))

#High Number indicate bad formations
Role_Percent['Ratio'] = Role_Percent.Season_Year_Yes_Percent/Role_Percent.Season_Year_No

#Drop unnecessary columns and clean up names
Role_Percent = Role_Percent[['Cov_Ret','Formation','Season_Year_Yes','Season_Year_No','Season_Year_Yes_Percent','Ratio']]
Role_Percent.columns = ['Cov_Ret','Formation','Concussion_Count','Not_Concussion_Percent','Concussion_Percent','Ratio']
display(Role_Percent)
#Fortunately I already have that data.  In hindsight, I should have looked at this first. 
RoleGroup_Ret = RoleGroup[(RoleGroup.Cov_Ret=='Return') & (RoleGroup.Group != 'Returner')]

#Very similar to above, but with a few tweaks
Role_Con = pd.merge(RoleGroup_Ret,video,on=['Season_Year','GameKey','PlayID'],how='left')
Role_Con.Concussion = Role_Con.Concussion.fillna('NO')

#Group by formation.  Separate formations by coverage and return teams
Role_Agg = Role_Con.groupby(['Group','Role','Concussion'])['Season_Year'].agg('count').reset_index()
Role_Agg.columns = ['Group','Role','Concussion','Count']
#Separate Yes and No's and join them to compare them.
#Many formations had no concussions.  We may revisit them later.
Role_Yes = Role_Agg[Role_Agg.Concussion=='YES']
Role_No = Role_Agg[Role_Agg.Concussion=='NO']

Role_Percent=pd.merge(Role_Yes,Role_No,on=['Group','Role'],suffixes=['_Yes','_No'],how='left')
Role_Percent['Count_Yes_Percent'] = Role_Percent.Count_Yes/len(video)
Role_Percent['Count_No_Percent'] = Role_Percent.Count_No/(len(play_information)-len(video))

#High Number indicate bad formations
Role_Percent['Ratio'] = Role_Percent.Count_Yes_Percent/Role_Percent.Count_No_Percent
Role_Percent=Role_Percent.rename(columns = {'Role':'Players'})
display(Role_Percent[['Group','Players','Count_Yes','Count_No','Ratio']])
#Calculate how players are lined up
#Based on above will only look at Dline and Linebacker
SidePlayer = player[player.Group.isin(['DLine','Linebacker'])]
SideGroup = SidePlayer.groupby(['Season_Year','GameKey','PlayID','Side']).agg({'Role': 'count'}).reset_index()
Side_Left = SideGroup[SideGroup.Side=='Left']
Side_Right = SideGroup[SideGroup.Side=='Right']
Side_Con = pd.merge(Side_Left,Side_Right,on=['Season_Year','GameKey','PlayID'],how='inner',suffixes=['_Left','_Right'])
Side_Con['Delta']= Side_Con.Role_Left-Side_Con.Role_Right
Side_Con = Side_Con[['Season_Year','GameKey','PlayID','Delta']]
Side_Con = pd.merge(Side_Con,video,on=['Season_Year','GameKey','PlayID'],how='left')
Side_Con.Concussion = Side_Con.Concussion.fillna('NO')

#Group by formation.  Separate formations by coverage and return teams
Side_Agg = Side_Con.groupby(['Delta','Concussion'])['Season_Year'].agg('count').reset_index()
Side_Agg.columns = ['Delta','Concussion','Count']
#Separate Yes and No's and join them to compare them.
#Many formations had no concussions.  We may revisit them later.
Side_Yes = Side_Agg[Side_Agg.Concussion=='YES']
Side_No = Side_Agg[Side_Agg.Concussion=='NO']

Side_Percent=pd.merge(Side_Yes,Side_No,on=['Delta'],suffixes=['_Yes','_No'],how='left')
Side_Percent['Count_Yes_Percent'] = Side_Percent.Count_Yes/len(video)
Side_Percent['Count_No_Percent'] = Side_Percent.Count_No/(len(play_information)-len(video))

#High Number indicate bad formations
Side_Percent['Ratio'] = Side_Percent.Count_Yes_Percent/Side_Percent.Count_No_Percent
display(Side_Percent[['Delta','Count_Yes','Count_No','Ratio']])
Side_Con['Balanced']=np.where(Side_Con.Delta==0,True,False)
Side_Agg = Side_Con.groupby(['Balanced','Concussion'])['Season_Year'].agg('count').reset_index()
Side_Agg.columns = ['Balanced','Concussion','Count']
#Separate Yes and No's and join them to compare them.
#Many formations had no concussions.  We may revisit them later.
Side_Yes = Side_Agg[Side_Agg.Concussion=='YES']
Side_No = Side_Agg[Side_Agg.Concussion=='NO']

Side_Percent=pd.merge(Side_Yes,Side_No,on=['Balanced'],suffixes=['_Yes','_No'],how='left')
Side_Percent['Count_Yes_Percent'] = Side_Percent.Count_Yes/len(video)
Side_Percent['Count_No_Percent'] = Side_Percent.Count_No/(len(play_information)-len(video))

#High Number indicate bad formations
Side_Percent['Ratio'] = Side_Percent.Count_Yes_Percent/Side_Percent.Count_No_Percent
display(Side_Percent[['Balanced','Count_Yes','Count_No','Ratio']])
#Combine the data to use
conc_player = pd.merge(video_review,player, left_on=['Season_Year','GameKey','PlayID','GSISID'],
                       right_on=['Season_Year','GameKey','PlayID','GSISID'],how='left')
conc_player = pd.merge(conc_player,play_information,on=['Season_Year','GameKey','PlayID'] ,how='left')
#conc_player.head()
display(conc_player.groupby('Player_Activity_Derived')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Evenly split between blocking and tackling')

display(conc_player.groupby('Primary_Impact_Type')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Most concussions involve hitting another helmet or body')

display(conc_player.groupby('Score_Diff_qcut')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('This is an unlikely variable, but I can get a sense of what random looks like for this dataset.')

display(conc_player.groupby('YardsforTD_qcut')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Very evenly split.  Interesting that we can see that 80% of returns are <15 yards.  Only 20% are "big" returns. ')

display(conc_player.groupby('NetYards_qcut')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Bad punt/punt coverage has slightly more concussions.')

display(conc_player.groupby('PuntYards_qcut')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('"Outkicking the coverage" is associated with long returns.  The longest punts are about equivalent for concussions.')

display(conc_player.groupby('Group')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Offensive Line has by far the most concussions. Second is Wing, which is a very similar position. One factor to take into account is that there are 5 OL, 2 Wings, 2 gunners, and 1 Returner on each punt.')

display(conc_player.groupby('Position')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Comparing this table with the previous table, and the obvious question is "What happened to the offensive line?"  Not a single offensive lineman received a concussion even though 14 players play o-line did.  Looking into this in more detailed revealed that to improve punt coverage, players play out of position on the offensive line.')

display(conc_player.groupby('Cov_Ret')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('3x injuries for the coverage team.  This finding was definitely unexpected. The difference is very large so should be significant even at these low numbers.')

display(conc_player.groupby('Side')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('No difference seen.')

display(conc_player.groupby('Season_Type')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('There are 4 preseason games and 16 regular season games.  On a per game basis, preseason has twice the concussions of regular season.')

display(conc_player.groupby('Season_Year')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Despite the 2017 rule change, no obvious progress')

display(conc_player.groupby('Friendly_Fire')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Most concussions do not involve friendly fire')

display(conc_player.groupby('Penalty')['GSISID'].count().reset_index().rename(columns = {'GSISID':'count'}))
print('Over a quarter of concussions are on penalty plays.  I will review the video to determine if the penalty and concussion are related. ')
NGS_Key = video[:][['Season_Year','GameKey','PlayID']]
# gets all csv with NGS in their filename
path = "../input/NFL-Punt-Analytics-Competition/"
NGS_csvs = [path+file for file in os.listdir(path) if 'NGS' in file]

NGS = pd.DataFrame() #initialize an empty dataframe

# loop to csv then appends it to df
for path_csv in NGS_csvs:
    _df = pd.read_csv(path_csv,low_memory=False)
    _df = pd.merge(NGS_Key,_df,how='left', on=['Season_Year','GameKey','PlayID'])
    NGS = NGS.append(_df,ignore_index=True)
    del _df # deletes the _df to free up memory
NGS = NGS.dropna(subset=['GSISID','x','y','dis','o','dir'])       
NGS['Time'] = pd.to_datetime(NGS.Time)
NGS.sort_values(['Season_Year','GameKey','PlayID','Time'])
NGS['Event'] = NGS.groupby(['Season_Year','GameKey','PlayID','Time'])['Event'].fillna(method='ffill')
NGS['Event'] = NGS.groupby(['Season_Year','GameKey','PlayID','Time'])['Event'].fillna(method='bfill')
def probplot(df,groupCol,varCol):
#if 1==1:
    #df = Tackle[:]
    #groupCol ='GameKey'
    #varCol='DIS_diff'
    g=df[[groupCol,varCol]].dropna()
    h=g.groupby(groupCol)[varCol].rank(pct=True,method='dense').reset_index()
    g=pd.merge(g,h,right_on='index',left_index=True,suffixes=('','_pct'))
    g=g.sort_values([groupCol,varCol])
    traces=[]
    for hue in sorted(g[groupCol].unique()):
            trace = go.Scatter(y=g[g[groupCol]==hue][varCol + '_pct'], x=g[g[groupCol]==hue][varCol],name=str(hue), showlegend=True)
            traces.append(trace)
    data = traces
    layout = go.Layout(
    autosize=False,
    width=500,
    height=500)
    fig = dict(data=data, layout=layout)
    #print("\n\n\t",play_description)
    #print(GameKey,PlayID)
    #print(URL)
    offline.iplot(fig)#, config=config) 
#Inner Join Player to remove players who were not on punt play and to remove players without NGS data
Tackle = pd.merge(NGS,player,on=['Season_Year','GameKey','PlayID','GSISID'],how='inner')
Tackle = pd.merge(NGS,video[['Season_Year','GameKey','PlayID','GSISID']], 
                             on=['Season_Year','GameKey','PlayID'],suffixes=['','_Conc'])

del NGS # deletes the _df to free up memory
#Run for Debugging
#GSISIDGroup = player.groupby(['GameKey','PlayID']).agg({'GSISID': 'count'}).reset_index()
#Note: GameKey=89, PLayID=4662 has 1 GSISID GameKey=319, PlayID=3019 has 16 GSISID

#Copy Concussed GSISID Data to a new column
Tackle['Conc_X'] = np.where(Tackle.GSISID==Tackle.GSISID_Conc,Tackle.x,np.nan)
Tackle['Conc_Y'] = np.where(Tackle.GSISID==Tackle.GSISID_Conc,Tackle.y,np.nan)
Tackle['Conc_Dis'] = np.where(Tackle.GSISID==Tackle.GSISID_Conc,Tackle.dis,np.nan)
Tackle['Conc_Dir'] = np.where(Tackle.GSISID==Tackle.GSISID_Conc,Tackle.dir,np.nan)

#Is there a better way to do this???
#Copy Conc_ position to all other positions.
#Note: calling ffill and then bfill on same line did not act as I expected.  Groupby only applied to first call
Tackle.Conc_X = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_X'].fillna(method='ffill')
Tackle.Conc_Y = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_Y'].fillna(method='ffill')
Tackle.Conc_X = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_X'].fillna(method='bfill')
Tackle.Conc_Y = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_Y'].fillna(method='bfill')
Tackle.Conc_Dis = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_Dis'].fillna(method='ffill')
Tackle.Conc_Dir = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_Dir'].fillna(method='ffill')
Tackle.Conc_Dis = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_Dis'].fillna(method='bfill')
Tackle.Conc_Dir = Tackle.groupby(['Season_Year','GameKey','PlayID','Time'])['Conc_Dir'].fillna(method='bfill')

#How Far is the player from the Conc_player  (Conc_player will always be 0)
Tackle['Distance']= ((Tackle.x-Tackle.Conc_X)**2+(Tackle.y-Tackle.Conc_Y)**2)**0.5

#Get DF in right order


Tackle.sort_values(['Season_Year','GameKey','PlayID','GSISID','Time'],inplace=True)
Tackle['Speed'] = Tackle.groupby(['Season_Year','GameKey','PlayID','GSISID'])['Distance'].diff(1)
Tackle['Acceleration'] = Tackle.groupby(['Season_Year','GameKey','PlayID','GSISID'])['Speed'].diff(1)


#Calculate beginning of each play
minSeconds = Tackle.groupby(['Season_Year','GameKey','PlayID'])['Time'].min().reset_index()
minSeconds.columns = ['Season_Year','GameKey','PlayID','PlayStart']

#Merge with Tackle DF and calculate seconds for each play
Tackle = pd.merge(Tackle,minSeconds,on=['Season_Year','GameKey','PlayID'],how='left')
Tackle['seconds'] = (Tackle['Time']-Tackle.PlayStart).dt.total_seconds()

#Sort again
Tackle.sort_values(['Season_Year','GameKey','PlayID','GSISID','seconds'],inplace=True)

#delta seconds & shift_seconds are used to address gaps in NGS data
#If there is a gap in the previous 2 records, then acceleration will be wrong.
Tackle['delta_seconds'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['seconds'].diff(1)
Tackle['shift_seconds'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['delta_seconds'].shift(1)


#Remove bad data rows
Tackle['Good_time'] = np.where((Tackle['delta_seconds']>0.15)|(Tackle['shift_seconds']>0.15),False,True)

Tackle['Conc_DIS_diff'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['Conc_Dis'].diff(1)
Tackle['Conc_DIS_diffA'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['Conc_DIS_diff'].shift(1)
Tackle['Conc_DIS_diffB'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['Conc_DIS_diff'].shift(-1)
Tackle['DIS_diff'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['dis'].diff(1)
Tackle['DIS_diffA'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['DIS_diff'].shift(1)
Tackle['DIS_diffB'] = Tackle.groupby(['GameKey','PlayID','GSISID'])['DIS_diff'].shift(-1)
Image("../input/images2/threegraphs.png")
probplot(Tackle,'GameKey','DIS_diff')

Tackle['Good_dis'] = np.where((Tackle.DIS_diff<-0.2) | (Tackle.DIS_diff>0.2) | 
                              (Tackle.DIS_diffA<-0.2) | (Tackle.DIS_diffA>0.2) | 
                              (Tackle.DIS_diffB<-0.2) | (Tackle.DIS_diffB>0.2) |
                              (Tackle.Conc_DIS_diff<-0.2) | (Tackle.Conc_DIS_diff>0.2) |
                              (Tackle.Conc_DIS_diffA<-0.2) | (Tackle.Conc_DIS_diffA>0.2) | 
                              (Tackle.Conc_DIS_diffB<-0.2) | (Tackle.Conc_DIS_diffB>0.2), False, True)
Tackle = Tackle[Tackle.Good_time & Tackle.Good_dis]
Tackle['Shift_Speed'] = Tackle['Speed'].shift(-1)
Tackle['Hit_Speed'] = np.where(((Tackle['Shift_Speed']/Tackle['Speed']<0) &
                               (Tackle['Distance']<5)),abs(Tackle['Speed']),0)
probplot(Tackle,'GameKey','DIS_diff')

#Add Partner where available
conc_player.Primary_Partner_GSISID = pd.to_numeric(conc_player.Primary_Partner_GSISID,errors='coerce') 
conc_pair = pd.merge(conc_player,player,how='left', left_on=['Season_Year','GameKey','PlayID','Primary_Partner_GSISID'],
                    right_on=['Season_Year','GameKey','PlayID','GSISID'], suffixes=('','_Partner'))

#Add in Jammers based on EDA
Role_Add = Role_Con[Role_Con.Group=='Jammer'][['Season_Year','GameKey','PlayID','Role']].rename(columns = {'Role':'Jammers'})
conc_pair = pd.merge(conc_pair,Role_Add, on=['Season_Year','GameKey','PlayID'], how='left')

#Add in Balance based on EDA
conc_pair = pd.merge(conc_pair,Side_Con[['Season_Year','GameKey','PlayID','Balanced']], on=['Season_Year','GameKey','PlayID'], how='left')

conc_pair = pd.merge(conc_pair,video[['Season_Year','GameKey','PlayID','URL']], on=['Season_Year','GameKey','PlayID'], how='left')
def load_layout():
    """
    Returns a dict for a Football themed Plot.ly layout 
    """
    layout = dict(
        title = "Player Activity",
        plot_bgcolor='darkseagreen',
        showlegend=False,
        width=640,
        height=400,
        margin = dict(t=1),
        xaxis=dict(
            autorange=False,
            range=[0, 120],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='',
            tickmode='array',
            tickvals=[10,20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            ticktext=['Goal', 10, 20, 30, 40, 50, 40, 30, 20, 10, 'Goal'],
            showticklabels=True
        ),
        yaxis=dict(
            title='',
            autorange=False,
            range=[-3.3,56.3],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='',
            showticklabels=False
        ),
        shapes=[
            dict(
                type='line',
                layer='below',
                x0=0,
                y0=0,
                x1=120,
                y1=0,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=0,
                y0=53.3,
                x1=120,
                y1=53.3,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=10,
                y0=0,
                x1=10,
                y1=53.3,
                line=dict(
                    color='white',
                    width=10
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=20,
                y0=0,
                x1=20,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=30,
                y0=0,
                x1=30,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=40,
                y0=0,
                x1=40,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=50,
                y0=0,
                x1=50,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=60,
                y0=0,
                x1=60,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),dict(
                type='line',
                layer='below',
                x0=70,
                y0=0,
                x1=70,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),dict(
                type='line',
                layer='below',
                x0=80,
                y0=0,
                x1=80,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=90,
                y0=0,
                x1=90,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),dict(
                type='line',
                layer='below',
                x0=100,
                y0=0,
                x1=100,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=110,
                y0=0,
                x1=110,
                y1=53.3,
                line=dict(
                    color='white',
                    width=10
                )
            )
        ]
    )
    return layout

end_where = ((Tackle.Event=='tackle') | 
             (Tackle.Event=='punt_downed') |
             (Tackle.Event=='out_of_bounds') |
             (Tackle.Event=='touchdown') |
             (Tackle.Event=='fair_catch'))
# Loading and plotting functions
def plot_play(play_number):
#play_number = 1
#if play_number!=-1:    
    """
    Plots player movements on the field for a given game, play, and two players
    """
    #print (Primary.loc[play_number])#[['Season_Year','Season_Type','Week','GameKey','PlayID'])
    GSISID = conc_pair.loc[play_number]['GSISID']
    GameKey = conc_pair.loc[play_number]['GameKey']
    PlayID = conc_pair.loc[play_number]['PlayID']
    play_description = conc_pair.iloc[play_number]["PlayDescription"]
    URL = conc_pair.iloc[play_number]["URL"]
    Second = conc_pair.loc[play_number]['GSISID_Partner']
    
    print ('Season_Type:', conc_pair.iloc[play_number]['Season_Type'],
           '  Season_Year:', conc_pair.iloc[play_number]['Season_Year'],
           '  GameKey:', conc_pair.iloc[play_number]['GameKey'],
           '  PlayID:', conc_pair.iloc[play_number]['PlayID'],
           '  Jammers:', conc_pair.iloc[play_number]['Jammers'],
           '  Balanced:', conc_pair.iloc[play_number]['Balanced'], 
           '  Penalty:', conc_pair.iloc[play_number]['Penalty'])
    display ()
    print ("GSISID:", GSISID,
               "  Role:", conc_pair.iloc[play_number]['Group'],
               "  Number:", conc_pair.iloc[play_number]['Number'],
               "  Position:", conc_pair.iloc[play_number]['Position'],
               "  Activity:", conc_pair.iloc[play_number]['Player_Activity_Derived'])
    if pd.isna(Second)==False:
        display()
        print ("Partner GSISID:", Second,
           "  Role:", conc_pair.iloc[play_number]['Group_Partner'],
           "  Number:", conc_pair.iloc[play_number]['Number_Partner'],
           "  Position:", conc_pair.iloc[play_number]['Position_Partner'],
           "  Activity:", conc_pair.iloc[play_number]['Primary_Partner_Activity_Derived'],
           "  Friendly Fire:", conc_pair.iloc[play_number]['Friendly_Fire'])
    display (play_description)

    game_df = Tackle[(Tackle.PlayID==PlayID) & (Tackle.GameKey==GameKey)].sort_values("Time")
    playstart = game_df[game_df.Event == "line_set"]["Time"].min()

    end_where = ((game_df.Event=='tackle') | 
             (game_df.Event=='punt_downed') |
             (game_df.Event=='out_of_bounds') |
             (game_df.Event=='touchdown') |
             (game_df.Event=='fair_catch'))
    
    playend = game_df[end_where]["Time"].min() + pd.to_timedelta(2, unit='s')
    
    game_df = game_df[(game_df.Time > playstart) & (game_df.Time < playend)]
    
    
    if len(game_df)==0:
          game_df = Tackle[(Tackle.PlayID==PlayID) & (Tackle.GameKey==GameKey)].sort_values("Time")
          #return URL
    #GameKey=str(pd.unique(game_df.GameKey)[0])
    #HomeTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[0]
    #VisitingTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[1]
    #YardLine = game_df[(game_df.PlayID==PlayID) & (game_df.GSISID==player1)]['YardLine'].iloc[0]
    
    traces=[]   
    game_df['Delta'] = game_df.Time - game_df.Time.min()
    game_df.Delta = game_df.Delta.dt.total_seconds()   
    
    playerid = int(GSISID)
    playernumber = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                              (player.GameKey==GameKey)]['Number'].values[0]
    playerGroup = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                              (player.GameKey==GameKey)]['Group'].values[0]
    trace = go.Scatter(
        x = game_df[game_df.GSISID==playerid].x,
        y = game_df[game_df.GSISID==playerid].y,
        name ='Position: '+str(playerGroup) + ' Number: '+str(playernumber),
        mode='markers',
        marker = dict(
        size = np.minimum(game_df[game_df.GSISID==playerid].Delta+6,10),
            color = 'rgba(255,255,0, .8)'))
    traces.append(trace)
    
    #Partner
    if pd.isna(Second)==False:
        playerid = int(Second)
        playernumber = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                                  (player.GameKey==GameKey)]['Number'].values[0]
        playerGroup = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                                  (player.GameKey==GameKey)]['Group'].values[0]
        trace = go.Scatter(
            x = game_df[game_df.GSISID==playerid].x,
            y = game_df[game_df.GSISID==playerid].y,
            name ='Position: '+str(playerGroup) + ' Number: '+str(playernumber),
            mode='markers',
            marker = dict(
            size = np.minimum(game_df[game_df.GSISID==playerid].Delta+6,10),
                color = 'rgba(0,255,0, .8)'))
        traces.append(trace)
    
    #get coverage    
    for playerid in pd.unique(player[(player.PlayID==PlayID) & (player.GameKey==GameKey) & 
                                    (player.Cov_Ret=='Coverage')]['GSISID']):
        playerid = int(playerid)
        playernumber = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                              (player.GameKey==GameKey)]['Number'].values[0]
        playerGroup = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                              (player.GameKey==GameKey)]['Group'].values[0]
        trace = go.Scatter(
            x = game_df[game_df.GSISID==playerid].x,
            y = game_df[game_df.GSISID==playerid].y,
            name ='Position: '+str(playerGroup) + ' Number: '+str(playernumber),
            mode='markers',
            marker = dict(
            size = 2,
            color = 'rgba(0, 0, 255, .8)'))
        traces.append(trace)
    
    #get receivers    
    for playerid in pd.unique(player[(player.PlayID==PlayID) & (player.GameKey==GameKey) & 
                                     (player.Cov_Ret=='Return')]['GSISID']):
        playerid = int(playerid)
        trace = go.Scatter(
            x = game_df[game_df.GSISID==playerid].x,
            y = game_df[game_df.GSISID==playerid].y,
            name ='Position: '+str(playerGroup) + ' Number: '+str(playernumber),
            mode='markers',
            marker = dict(
            size = 2,
            color = 'rgba(255,0,0, .8)'))
        traces.append(trace)

    
    layout = load_layout()
   # layout['title'] =  HomeTeam + \
   # ' vs. ' + VisitingTeam + \
   # '<br>Possession: ' + \
   # YardLine.split(" ")[0] +'@'+YardLine.split(" ")[1]
    data = traces
    fig = dict(data=data, layout=layout)
    #print("\n\n\t",play_description)
    #print(GameKey,PlayID)
    #print(URL)
    offline.iplot(fig, config=config)
    
    return URL
    #HTML('<video width="560" height="315" controls> <source src=a type="video/mp4"></video>')

# Loading and plotting functions
def plot_acceleration(play_number):
#play_number=16
#if play_number!=-1:
    GSISID = conc_pair.loc[play_number]['GSISID']
    GameKey = conc_pair.loc[play_number]['GameKey']
    PlayID = conc_pair.loc[play_number]['PlayID']   
    
    game_df = Tackle[(Tackle.PlayID==PlayID) & (Tackle.GameKey==GameKey)].sort_values("Time")
    playstart = game_df[game_df.Event == "line_set"]["Time"].min()

    end_where = ((game_df.Event=='tackle') | 
             (game_df.Event=='punt_downed') |
             (game_df.Event=='out_of_bounds') |
             (game_df.Event=='touchdown') |
             (game_df.Event=='fair_catch'))
    
    playend = game_df[end_where]["Time"].min() + pd.to_timedelta(2, unit='s')
    
    game_df = game_df[(game_df.Time > playstart) & (game_df.Time < playend)] 
    
    if len(game_df)==0:
          game_df = Tackle[(Tackle.PlayID==PlayID) & (Tackle.GameKey==GameKey)].sort_values("Time")
    game_df['Delta'] = game_df.Time - game_df.Time.min()
    game_df.Delta = game_df.Delta.dt.total_seconds()
    Fast_Hits = game_df.groupby('GSISID')['Hit_Speed'].max().reset_index()
    Fast_Hits = Fast_Hits[Fast_Hits.Hit_Speed> 0.1]
    if len(Fast_Hits)==0:
        print("No Fast Hits")
        return
    traces=[]  
    for playerid in pd.unique(Fast_Hits.GSISID):
        playerid = int(playerid)
        playernumber = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                              (player.GameKey==GameKey)]['Number'].values[0]
        playerGroup = player[(player.GSISID==playerid)&(player.PlayID==PlayID) & 
                              (player.GameKey==GameKey)]['Group'].values[0]

        trace = go.Scatter(
            x = game_df[game_df.GSISID==playerid].Delta,
            y = game_df[game_df.GSISID==playerid].Hit_Speed*36000/1760,
            name ='Position: '+str(playerGroup) + ' Number: '+str(playernumber),
            mode='lines+markers',
            marker = dict(
            size = 10
            #,color = 'rgba(152, 0, 0, .8)'
            ))
        traces.append(trace)
    layout = go.Layout(
    title='Hit Speed vs Time',
        width=800,
        height=350,
        showlegend=True,
    yaxis=dict(
            title='Hit Speed (MPH)'
            )
        )
    fig = go.Figure(data=traces,layout=layout)
    offline.iplot(fig)#, config=config)
    return
    
    #HTML('<video width="560" height="315" controls> <source src=a type="video/mp4"></video>')

#for play_no in range(0,36):
play_no=12
URL=plot_play(play_no)
plot_acceleration(play_no)
print(URL)
HTML("""<video width="840" height="350" controls=""><source src="{0}"> type="video/mp4"</video>""".format(URL))
SpeedChart = pd.merge(Tackle[(Tackle.Hit_Speed>0.4)],conc_pair[['Season_Year','GameKey','PlayID','Primary_Partner_GSISID']],
                      how='left',left_on=['Season_Year','GameKey','PlayID','GSISID'],
                       right_on=['Season_Year','GameKey','PlayID','Primary_Partner_GSISID'])
SpeedChart['Partner'] = np.where(pd.isna(SpeedChart.Primary_Partner_GSISID),'NotPartner','Partner')
SpeedChart['Hit_Speed'] = SpeedChart['Hit_Speed']*36000/1760 
probplot(SpeedChart,'Partner','Hit_Speed')
bad = pd.read_csv('../input/extradata/Existing_Penalties.csv')
display(bad)
Explained = pd.merge(bad,conc_player,on=['Season_Year','GameKey','PlayID'],how='right')
print('Table shows specific reason each play was excluded from further analysis')
Explained=Explained[pd.isna(Explained.Reason)]
def Display_Explained(strColumn):
#if 1==1:
    #strColumn = 'Player_Activity_Derived'
    clean = Explained.groupby(strColumn)['GSISID'].count().reset_index().rename(columns = {'GSISID':'Clean'})
    data = conc_player.groupby(strColumn)['GSISID'].count().reset_index().rename(columns = {'GSISID':'Raw'})
    merged = pd.merge(clean,data)
    merged['Removed']=merged.Clean-merged.Raw
    display(merged)
    
Display_Explained('Player_Activity_Derived')
print('Tackling is the biggest remaining issue. Looking at videos, the main problem was multiple players tackling. This situation caused players heads to bump into each other.  I can not think of a rule that would prevent group tackles and maintain the integrity of the game.')

Display_Explained('Primary_Impact_Type')
print('Helmet-to-helmet hits were the most reduced by rules changes')

Display_Explained('YardsforTD_qcut')
print('Almost all concussions occur when the punter is punting for distance')

Display_Explained('Cov_Ret')
print('The initial analysis of 3x injuries for the coverage team is unchanged.')

Display_Explained('Season_Type')
print('There problem with preseason games appears even worse after cleaning the data. Remember there are 4x more regular season games than preseason games')

Display_Explained('Season_Year')
print('2017 now looks much worse than 2016 but the numbers are so low the difference is not meaningful')

Display_Explained('Penalty')
print('5 concussions occurred on plays that were nullified for other penalties.')
#Calculate how players are lined up
#Based on above will only look at Dline and Linebacker
SidePlayer = player[player.Group.isin(['DLine','Linebacker'])]
SideGroup = SidePlayer.groupby(['Season_Year','GameKey','PlayID','Side']).agg({'Role': 'count'}).reset_index()
SideGroup = pd.merge(bad,SideGroup,on=['Season_Year','GameKey','PlayID'],how='right')
SideGroup = SideGroup[pd.isna(SideGroup.Reason)]
Side_Left = SideGroup[SideGroup.Side=='Left']
Side_Right = SideGroup[SideGroup.Side=='Right']
Side_Con = pd.merge(Side_Left,Side_Right,on=['Season_Year','GameKey','PlayID'],how='inner',suffixes=['_Left','_Right'])
Side_Con['Delta']= Side_Con.Role_Left-Side_Con.Role_Right
Side_Con = Side_Con[['Season_Year','GameKey','PlayID','Delta']]
Side_Con = pd.merge(Side_Con,video,on=['Season_Year','GameKey','PlayID'],how='left')
Side_Con.Concussion = Side_Con.Concussion.fillna('NO')

#Group by formation.  Separate formations by coverage and return teams
Side_Agg = Side_Con.groupby(['Delta','Concussion'])['Season_Year'].agg('count').reset_index()
Side_Agg.columns = ['Delta','Concussion','Count']
#Separate Yes and No's and join them to compare them.
#Many formations had no concussions.  We may revisit them later.
Side_Yes = Side_Agg[Side_Agg.Concussion=='YES']
Side_No = Side_Agg[Side_Agg.Concussion=='NO']

Side_Percent=pd.merge(Side_Yes,Side_No,on=['Delta'],suffixes=['_Yes','_No'],how='left')
Side_Percent['Count_Yes_Percent'] = Side_Percent.Count_Yes/len(video)
Side_Percent['Count_No_Percent'] = Side_Percent.Count_No/(len(play_information)-len(video))

#High Number indicate bad formations
Side_Percent['Ratio'] = Side_Percent.Count_Yes_Percent/Side_Percent.Count_No_Percent
Side_Percent
Side_Con['Balanced']=np.where(Side_Con.Delta==0,True,False)
Side_Agg = Side_Con.groupby(['Balanced','Concussion'])['Season_Year'].agg('count').reset_index()
Side_Agg.columns = ['Balanced','Concussion','Count']
#Separate Yes and No's and join them to compare them.
#Many formations had no concussions.  We may revisit them later.
Side_Yes = Side_Agg[Side_Agg.Concussion=='YES']
Side_No = Side_Agg[Side_Agg.Concussion=='NO']

Side_Percent=pd.merge(Side_Yes,Side_No,on=['Balanced'],suffixes=['_Yes','_No'],how='left')
Side_Percent['Count_Yes_Percent'] = Side_Percent.Count_Yes/len(Explained)
Side_Percent['Count_No_Percent'] = Side_Percent.Count_No/(len(play_information)-len(Explained))

#High Number indicate bad formations
Side_Percent['Ratio'] = Side_Percent.Count_Yes_Percent/Side_Percent.Count_No_Percent
Side_Percent[['Balanced','Count_Yes','Count_No','Ratio']]
#Fortunately I already have that data.  In hindsight, I should have looked at this first. 
RoleGroup_Ret = RoleGroup[(RoleGroup.Cov_Ret=='Return') & (RoleGroup.Group != 'Returner')]
RoleGroup_Ret= pd.merge(bad,RoleGroup_Ret,on=['Season_Year','GameKey','PlayID'],how='right')
RoleGroup_Ret = RoleGroup_Ret[pd.isna(RoleGroup_Ret.Reason)]
#Very similar to above, but with a few tweaks
Role_Con = pd.merge(RoleGroup_Ret,video,on=['Season_Year','GameKey','PlayID'],how='left')
Role_Con.Concussion = Role_Con.Concussion.fillna('NO')

#Group by formation.  Separate formations by coverage and return teams
Role_Agg = Role_Con.groupby(['Group','Role','Concussion'])['Season_Year'].agg('count').reset_index()
Role_Agg.columns = ['Group','Role','Concussion','Count']
#Separate Yes and No's and join them to compare them.
#Many formations had no concussions.  We may revisit them later.
Role_Yes = Role_Agg[Role_Agg.Concussion=='YES']
Role_No = Role_Agg[Role_Agg.Concussion=='NO']

Role_Percent=pd.merge(Role_Yes,Role_No,on=['Group','Role'],suffixes=['_Yes','_No'],how='left')
Role_Percent['Count_Yes_Percent'] = Role_Percent.Count_Yes/len(Explained)
Role_Percent['Count_No_Percent'] = Role_Percent.Count_No/(len(play_information)-len(Explained))

#High Number indicate bad formations
Role_Percent['Ratio'] = Role_Percent.Count_Yes_Percent/Role_Percent.Count_No_Percent
Role_Percent[Role_Percent.Group=='Jammer'][['Role','Count_Yes','Count_No','Ratio']].rename(columns = {'Role':'Jammers'})
print('Short Returns on Distance Punts')
good = pd.merge(bad,Role_Con[(Role_Con.Group == "Jammer") & (Role_Con.Concussion=="YES")], 
                on=['Season_Year','GameKey','PlayID'],how='right')
good = good[(pd.isna(good.Reason_x))]
good = pd.merge(good,play_information[(play_information.YardsforTD>65)] , on=['Season_Year','GameKey','PlayID'],how='left')
good = good[~(pd.isna(good.Season_Type))&(good.ReturnYards<10)]
display(good.groupby('Role')['ReturnYards'].agg(['mean','count']).reset_index().rename(columns = {'Role':'Jammers', 'mean':'Return Yards'}).dropna())
#display(good.groupby('Role')['PuntYards'].agg(['mean','count']).reset_index().rename(columns = {'Role':'Jammers', 'mean':'Punt Yards'}).dropna())

good = pd.merge(Role_Con[Role_Con.Group=='Jammer'],play_information[(play_information.YardsforTD>65)], on=['Season_Year','GameKey','PlayID'])
good = good[(good.Role>1)&(good.ReturnYards<10)]
display(good.groupby('Role')['ReturnYards'].agg(['mean','count']).reset_index().rename(columns = {'Role':'Jammers', 'mean':'Return Yards'}).dropna())
print('On short returns, more jammers do not appear to have an impact on concussions, but the numbers are very small')
print('')
print('Long Returns on Distance Punts')
good = pd.merge(bad,Role_Con[(Role_Con.Group == "Jammer") & (Role_Con.Concussion=="YES")], 
                on=['Season_Year','GameKey','PlayID'],how='right')
good = good[(pd.isna(good.Reason_x))]
good = pd.merge(good,play_information[(play_information.YardsforTD>65)] , on=['Season_Year','GameKey','PlayID'],how='left')
good2 = good[:]
good = good[~(pd.isna(good.Season_Type))&(good.ReturnYards>10)]
display(good.groupby('Role')['PuntYards'].agg(['mean','count']).reset_index().rename(columns = {'Role':'Jammers', 'mean':'Punt Yards'}).dropna())

good = pd.merge(Role_Con[Role_Con.Group=='Jammer'],play_information[(play_information.YardsforTD>65)], on=['Season_Year','GameKey','PlayID'])
good = good[(good.Role>1)&(good.ReturnYards>10)]
display(good.groupby('Role')['ReturnYards'].agg(['mean','count']).reset_index().rename(columns = {'Role':'Jammers', 'mean':'Return Yards'}).dropna())
print('On long returns, more jammers appear to have an impact on concussions, but the numbers are very small')

good2 = good2[(good2.Role>1)]
display(good2.groupby('Role')['ReturnYards'].agg(['mean','count']).reset_index().rename(columns = {'Role':'Jammers', 'mean':'Return Yards'}).dropna())
print('Looking at all of the concussion data, there is a much larger difference in return yards than would be expected')
probplot(good,'Role','ReturnYards')
probplot(good2,'Role','ReturnYards')
down = len(play_information[play_information.Downed==1])
touchback = len(play_information[play_information.Touchback==1])
OOB = len(play_information[play_information.OutofBounds==1])
faircatch = len(play_information[play_information.FairCatch==1])
returns = len(play_information.ReturnYards.dropna())
punts = len(play_information)
print ('Downed: {:.1%} Touchback: {:.1%} OutofBounds: {:.1%} Fair Catch: {:.1%} Returns: {:.1%}'.format(down/punts,touchback/punts,OOB/punts,faircatch/punts,returns/punts))
long_punts = play_information[(play_information.YardsforTD>65)]
#long_punts.ReturnYards = long_punts.ReturnYards.fillna(0)
long_punts = long_punts.dropna(subset=['YardsforTD'])
long_punts['ten'] = np.where((long_punts['ReturnYards']<10) | (long_punts.FairCatch==1),1,0)
long_punts['five'] = np.where((long_punts['ReturnYards']<5) | (long_punts.FairCatch==1),1,0)
display(long_punts.groupby('FairCatch')['NetYards'].mean().reset_index())
display(long_punts.groupby('FairCatch')['PuntYards'].mean().reset_index())
print('Fair catches occur on shorter punts but result in fewer net yards')
display(long_punts.groupby('five')['NetYards'].mean().reset_index())
print('A five yard bonus results in fair catches and not fair catches having the same net yards')
display(long_punts.groupby('ten')['NetYards'].mean().reset_index())
print('A ten yard bonus makes fair catches better than not fair catches.')
display(long_punts.groupby('OutofBounds')['NetYards'].mean().reset_index())
print('Kicking out of bounds is slightly better than not, but not as good as forcing a fair catch')
print('Average Return Yards: ', play_information.ReturnYards.mean())
