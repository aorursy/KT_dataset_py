import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import stats
import networkx as nx
import random
import seaborn as sns
nc=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv",encoding='utf-8')
nc["sum_score"]=nc.WScore+nc.LScore
nc["score_distance"]=nc.WScore-nc.LScore
n_points=pd.DataFrame()
group_by=nc.groupby(by=["Season"])
n_points["avr_sumscore"]=group_by["sum_score"].mean()
n_points["avr_scoredistance"]=group_by["score_distance"].mean()
n_points["Season"]=range(1985,2020)
rc=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv",encoding='utf-8')
rc["sum_score"]=rc.WScore+rc.LScore
rc["score_distance"]=rc.WScore-rc.LScore
r_points=pd.DataFrame()
group_by=rc.groupby(by=["Season"])
r_points["avr_sumscore"]=group_by["sum_score"].mean()
r_points["avr_scoredistance"]=group_by["score_distance"].mean()
r_points["Season"]=range(1985,2020)

plt.figure(figsize=(16,8))
plt.xlim(1984,2020)
plt.xticks(np.arange(1985, 2020, 1))
plt.title("Annual average sum points of games")
plt.xlabel("Season")
plt.ylabel("Points")
plt.plot(n_points["Season"],n_points["avr_sumscore"],label="Mad March",color='blue')
plt.scatter(n_points["Season"],n_points["avr_sumscore"],color='blue')
plt.plot(r_points["Season"],r_points["avr_sumscore"],label="Regular",color='red')
plt.scatter(r_points["Season"],r_points["avr_sumscore"],color='red')
plt.legend(["March Madness","Regular"])
plt.grid(c='gray')
plt.show()
plt.figure(figsize=(16,8))
plt.xlim(1984,2020)
plt.xticks(np.arange(1985, 2020, 1))
plt.title("Annual average score differential")
plt.xlabel("Season")
plt.ylabel("Score Differential")
plt.plot(n_points["Season"],n_points["avr_scoredistance"],label="Mad March",color='blue')
plt.scatter(n_points["Season"],n_points["avr_scoredistance"],color='blue')
plt.plot(r_points["Season"],r_points["avr_scoredistance"],label="Regular",color='red')
plt.scatter(r_points["Season"],r_points["avr_scoredistance"],color='red')
plt.legend(["March Madness","Regular"])
plt.grid(c='black')
plt.show()
teaminfo=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv",encoding='utf-8')
def outlierteams(yearlist):
    fig = plt.figure(figsize=(36,6))
    i=1
    for item in yearlist:
        nc_year=nc[nc.Season==item]
        nc_year=nc_year[["WTeamID","LTeamID","score_distance"]]
        rc_year=rc[rc.Season==item]
        rc_year=rc_year[["WTeamID","LTeamID","score_distance"]]
        ax1 = fig.add_subplot(1,9,i)
        ax1.grid(axis='both',c='white')
        ax1.set_ylabel("Score Distance")
        ax1.set_title("Score distance in "+str(item))
        ax1.boxplot([[x for x in list(nc_year.score_distance)],[x for x in list(rc_year.score_distance)]],showfliers=False,showmeans=True,widths = 0.6)
        ax1.set_xlabel("March Madness     Regular")
        ax1.set_xticks([])
        i=i+1
    plt.show()
outlierteams([1993,1996,1999,2001,2008,2009,2013,2016,2019])
ncaadetail=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv",encoding='utf-8')
ncaadetail["sumpoints"]=ncaadetail.WScore+ncaadetail.LScore
ncaadetail["sumrebounds"]=ncaadetail.WOR+ncaadetail.WDR+ncaadetail.LOR+ncaadetail.LDR
ncaadetail["sumblocks"]=ncaadetail.WBlk+ncaadetail.LBlk
ncaadetail["sumsteals"]=ncaadetail.WStl+ncaadetail.LStl
ncaadetail["sumast"]=ncaadetail.WAst+ncaadetail.LAst

avr_ncaa=pd.DataFrame()
group_by=ncaadetail.groupby(by=["Season"])
avr_ncaa["avr_point"]=group_by["sumpoints"].mean()
avr_ncaa["avr_rebound"]=group_by["sumrebounds"].mean()
avr_ncaa["avr_block"]=group_by["sumblocks"].mean()
avr_ncaa["avr_steal"]=group_by["sumsteals"].mean()
avr_ncaa["avr_ast"]=group_by["sumast"].mean()
avr_ncaa["Season"]=range(2003,2020)
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
avr_ncaa["avr_point"]=avr_ncaa[['avr_point']].apply(max_min_scaler)
avr_ncaa["avr_rebound"]=avr_ncaa[['avr_rebound']].apply(max_min_scaler)
avr_ncaa["avr_block"]=avr_ncaa[['avr_block']].apply(max_min_scaler)
avr_ncaa["avr_steal"]=avr_ncaa[['avr_steal']].apply(max_min_scaler)
avr_ncaa["avr_ast"]=avr_ncaa[['avr_ast']].apply(max_min_scaler)


feature = np.array([u'points', u'rebounds', u'blocks',u'steals',u'assist']) 
nAttr=5
i=1
fig=plt.figure(figsize=(24,20),facecolor="white")
plt.axis('off')
plt.grid()
for item in avr_ncaa.iterrows():
    values=np.array(list(item[1])[:-1])
    angles=np.linspace(0,2*np.pi,nAttr,endpoint=False)
    values=np.concatenate((values,[values[0]]))
    angles=np.concatenate((angles,[angles[0]]))  
    ax1 = fig.add_subplot(4,5,i,polar=True)
    ax1.set_title('Year '+str(int(item[1][-1])),loc="left")
    ax1.plot(angles,values,'bo-',color='g',linewidth=2)
    ax1.fill(angles,values,facecolor='g',alpha=0.2)
    ax1.set_thetagrids(angles * 180/np.pi, feature)
    ax1.grid(True)
    ax1.set_theta_zero_location(loc='E') 
    i=i+1
ncaadetail["sumgoals"]=ncaadetail.WFGM+ncaadetail.LFGM+ncaadetail.WFGM3+ncaadetail.LFGM3+ncaadetail.WFTM+ncaadetail.LFTM
ncaadetail["sumattempts"]=ncaadetail.WFGA+ncaadetail.LFGA+ncaadetail.WFGA3+ncaadetail.LFGA3+ncaadetail.WFTA+ncaadetail.LFTA
ncaadetail["3pointers"]=ncaadetail.WFGM3+ncaadetail.LFGM3
ncaadetail["3pointersattempts"]=ncaadetail.WFGA3+ncaadetail.LFGA3
ncaadetail["sumast"]=ncaadetail.WAst+ncaadetail.LAst
group_by=ncaadetail.groupby(by=["Season"])
avr_ncaa["avr_ast"]=group_by["sumast"].mean()
avr_ncaa["avr_goal"]=group_by["sumgoals"].mean()
avr_ncaa["avr_attempts"]=group_by["sumattempts"].mean()
avr_ncaa["avr_3pointers"]=group_by["3pointers"].mean()
avr_ncaa["avr_3pointersattempts"]=group_by["3pointersattempts"].mean()
avr_ncaa["avr_3pointersratio"]=avr_ncaa["avr_3pointers"]/avr_ncaa["avr_3pointersattempts"]
avr_ncaa["Season"]=range(2003,2020)
avr_ncaa["avr_astratio"]=avr_ncaa["avr_ast"]/avr_ncaa["avr_goal"]
avr_ncaa["avr_3pointsratio"]=avr_ncaa["avr_3pointersattempts"]/avr_ncaa["avr_attempts"]

plt.figure(figsize=(10,8))
plt.xlim(2002,2020)
my_x_ticks = np.arange(2002, 2020, 1)
plt.xticks(my_x_ticks)
plt.ylim(0,120)
plt.title("Annual average sum of ast and goals of NCAA")
plt.xlabel("Season")
plt.ylabel("Number of goals/asts")
plt.bar(x=list(avr_ncaa["Season"]),height=list(avr_ncaa["avr_goal"]),label="goals",width=0.4)
plt.bar(x=list(avr_ncaa["Season"]),height=list(avr_ncaa["avr_ast"]),label="ast",width=0.4)
plt.legend(["goals","ast"],loc = 'upper left')
plt.grid()
plt.show()
plt.figure(1,figsize=(20,16))
plt.subplot(121)
plt.xticks(np.arange(0, 60, 5))
plt.yticks(np.arange(2002, 2020, 1))
plt.ylim(2002,2020)
plt.title("Annual average 3-pointer made and attempts of NCAA")
plt.ylabel("Season")
plt.xlabel("Number of 3-pointer made/attempts")
plt.barh(list(avr_ncaa["Season"]),list(avr_ncaa["avr_3pointersattempts"]),label="3-pointer attempts")
plt.barh(list(avr_ncaa["Season"]),list(avr_ncaa["avr_3pointers"]),label="3-pointer made")
plt.legend(["attempts","3 points"],loc = 'upper left')
plt.subplot(122)
plt.xlim(2002,2020)
plt.yticks(np.arange(0, 0.3,0.05))
plt.xticks(np.arange(2002, 2020,1))
plt.ylim(0,0.3)
plt.title("Ratio of 3-poniters attempts")
plt.xlabel("Season")
plt.ylabel("Ratio of 3-pointers attempts")
plt.bar(list(avr_ncaa["Season"]),list(avr_ncaa["avr_3pointsratio"]),label="attempts")
plt.show()
gamecity=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MGameCities.csv",encoding='utf-8')
city_state=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/Cities.csv",encoding='utf-8')
gamecity=pd.merge(gamecity,city_state,"left",on="CityID")
gamecityncaa=gamecity[gamecity.CRType=="NCAA"].copy()
gamecityncaa["id"]=range(1,668)
games_state=pd.DataFrame()
games_state=gamecityncaa.groupby(by=["State"])["id"].count().reset_index(name="count")

plt.figure(figsize=(15,12))
plt.bar(games_state["State"],games_state["count"],color='red')
plt.title("Races State Distribution")
plt.axhline(y=games_state["count"].mean(),color="blue")
plt.axhline(y=games_state["count"].max(),color="yellow")
plt.ylabel("Num of Games Hold",fontsize=15)
plt.xlabel("State",fontsize=15)
plt.annotate('average: 19', xy=(10,19.5), xytext=(8,25),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('max: 70', xy=(7,70), xytext=(4,65),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.grid(c='gray')
plt.show()
ncaacompacttop4=nc[nc.DayNum==152].copy()
top4=pd.DataFrame()
year=[]
teams=[]
for row in ncaacompacttop4.iterrows():
    year.append(row[1][0])
    year.append(row[1][0])
    teams.append(row[1][2])
    teams.append(row[1][4])

top4["Season"]=year
top4["TeamID"]=teams

teaminfo=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv",encoding='utf-8')
top4=pd.merge(top4,teaminfo,"left", on="TeamID")
top4teams=top4.groupby(["TeamName"])["TeamID"].count().reset_index(name="count")
top4teams=top4teams[top4teams["count"]>=4]

plt.figure(figsize=(10,8))
plt.xlim(0,15)
plt.title("Strong Teams")
plt.ylabel("Teams")
plt.xlabel("Final Four times")
plt.barh(list(top4teams["TeamName"]),list(top4teams["count"]))
plt.show()
teaminfo["age"]=teaminfo.LastD1Season-teaminfo.FirstD1Season
teamage=teaminfo.groupby(["age"])["TeamID"].count().reset_index(name="count")

teamage['panduan'] = teamage['age'].apply(lambda x: 1 if x==35 else 0)
teamage=teamage.groupby(["panduan"])["count"].sum().reset_index(name="count")
teamage.panduan=["Less than 35 years","35 years"]

plt.figure(figsize=(10,8))
plt.pie(teamage['count'],labels=teamage.panduan,autopct='%1.1f%%',startangle=150,explode=[0.1,0],shadow=True)
plt.legend(loc="upper right",fontsize=10,borderaxespad=0.3)
plt.title("Team age")
plt.axis('equal')
plt.show()  
events2015=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2015.csv",encoding='utf-8')
events2016=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2016.csv",encoding='utf-8')
events2017=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2017.csv",encoding='utf-8')
events2018=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2018.csv",encoding='utf-8')
events2019=pd.read_csv(f"../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2019.csv",encoding='utf-8')
events=pd.concat([events2015,events2016,events2017,events2018,events2019])
events_hasarea=events[events.Area!=0]
columns=["foul","made2","made3","miss2","miss3","turnover"]
events_issues=pd.DataFrame(index=range(1,14),columns=[])
for item in columns:
    events_issue=events_hasarea[events_hasarea.EventType==item]
    events_issue=pd.DataFrame(events_issue.groupby(by=["Area"]).size(),columns=[item])
    events_issues=pd.concat([events_issues,events_issue],sort=True,axis=1)
    
events_issues=events_issues.apply(lambda x: round((x - np.min(x)) / (np.max(x) - np.min(x)),2))
events_issues=events_issues.fillna(0)

fig=plt.figure(figsize=(6,12))
ax = fig.add_subplot(111)
im = ax.imshow(events_issues, cmap=plt.cm.summer)
ax.xaxis.set_ticks_position('top')
ax.set_xticks(np.arange(len(events_issues.columns)))
ax.set_yticks(np.arange(len(events_issues.iloc[:,0])))
ax.set_xticklabels(events_issues.columns)
ax.set_yticklabels(events_issues.index) 
bottom, top = ax.get_ylim()
ax.set_ylim(bottom, top)
fig.colorbar(im,pad=0.03) 
ax.set_title("Heatmap for areas and actions",fontsize=16) 
plt.grid()
plt.show()
regular_19=rc[rc.Season==2019]
regular_19=regular_19.loc[:,["Season",'DayNum','WTeamID','LTeamID']]
regular_19_net=pd.DataFrame(regular_19.groupby(by=['WTeamID','LTeamID']).size(),columns=["num_of_games"])
allwteams=list(set(regular_19.WTeamID))
alllteams=list(set(regular_19.LTeamID))
allteams=list(set(allwteams+alllteams))
lst=[]
nodes=[]
for item in regular_19_net.iterrows():
    if item[1][0]>=2:
        lst.append((item[0][0],item[0][1],item[1][0]))
        nodes.append(item[0][0])
        nodes.append(item[0][1])
nodes=list(set(nodes))
G=nx.Graph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(lst)
plt.figure(figsize=(20,12))             
plt.subplot(121)           
nx.draw_networkx(G,pos=nx.kamada_kawai_layout(G),with_labels=False,edge_color='black',node_size=10,alpha=0.5,node_shape='s',width=0.4)
plt.title("In Conference games")
plt.legend(["Team"],loc='upper left')

lst=[]
nodes=[]
for item in regular_19_net.iterrows():
    if item[1][0]>=0:
        lst.append((item[0][0],item[0][1],item[1][0]))
        nodes.append(item[0][0])
        nodes.append(item[0][1])
nodes=list(set(nodes))
G=nx.Graph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(lst)
plt.subplot(122)               
nx.draw_networkx(G,pos=nx.kamada_kawai_layout(G),with_labels=False,edge_color='black',node_size=10,alpha=0.5,node_shape='s',width=0.2)
plt.legend(["Team"],loc='upper left')
plt.title("All games")
plt.show()
degree=nx.degree_histogram(G)
x=range(len(degree))
y=[z/float(sum(degree))for z in degree]
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree-distribution")
plt.loglog(x,y,color="blue",linewidth=2)
plt.show()
dac=nx.degree_assortativity_coefficient(G)
x=list(G.degree())
y=nx.average_neighbor_degree(G)
x=[item[1] for item in x]
y=[item[1] for item in y.items()]
plt.xlabel("Degree")
plt.ylabel("Average-neighbor degree")
plt.title("degree_assortativity")
plt.scatter(x,y,color="blue",s=1)
plt.show()
print("degree assortativity coefficient of the network："+str(dac))
cc=nx.average_clustering(G)
asd=nx.average_shortest_path_length(G,weight='yes')

cc_random=[]
asd_random=[]
for i in range(50):
    lst1=[(a,b,random.randint(1,6) if random.choice([0,1]) else 0) for a,b,c in lst]
    lst1=[item for item in lst1 if item[2]!=0]
    G_random=nx.Graph()
    G_random.add_nodes_from(nodes)
    G_random.add_weighted_edges_from(lst1)
    cc_random.append(nx.average_clustering(G_random))
    asd_random.append(nx.average_shortest_path_length(G_random,weight='yes'))

plt.figure(figsize=(9,7))
plt.subplot(121)
plt.boxplot(cc_random)
plt.scatter(x=1,y=cc,color='blue',s=2)
plt.annotate("NCAA 2019",xy=(1,cc),xytext=(0.5,cc-0.03),arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlim()
plt.grid(c='white')
plt.ylabel("clustering coefficients")

plt.subplot(122)
plt.boxplot(asd_random)
plt.scatter(x=1,y=asd,color='red',s=2)
plt.annotate("NCAA 2019",xy=(1,asd),xytext=(1.3,asd+0.1),arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlim()
plt.ylabel("average shortest distance")
plt.grid(c='white')
plt.show()
def time_with_newedges(years):
    results=[]  
    dict_teamrivals={}
    for year in years:
        regular_year=rc[rc.Season==year]
        regular_year=regular_year.loc[:,["Season",'DayNum','WTeamID','LTeamID']]
        lastyear=year-1
        regular_lastyear=rc[rc.Season==lastyear]
        regular_lastyear=regular_lastyear.loc[:,["Season",'DayNum','WTeamID','LTeamID']]


        allwteams=list(set(regular_year.WTeamID))
        alllteams=list(set(regular_year.LTeamID))
        allteams=list(set(allwteams+alllteams))
        num_teams_year=len(allteams)


        regular_year_net=pd.DataFrame(regular_year.groupby(by=['WTeamID','LTeamID']).size(),columns=["num_of_games"])
        lst=[]
        nodes=[]
        for item in regular_year_net.iterrows():
            lst.append((item[0][0],item[0][1],item[1][0]))
            nodes.append(item[0][0])
            nodes.append(item[0][1])
        nodes=list(set(nodes))
        G1=nx.Graph()
        G1.add_nodes_from(nodes)
        G1.add_weighted_edges_from(lst)


        regular_lastyear_net=pd.DataFrame(regular_lastyear.groupby(by=['WTeamID','LTeamID']).size(),columns=["num_of_games"])
        lst=[]
        nodes=[]
        for item in regular_lastyear_net.iterrows():
            lst.append((item[0][0],item[0][1],item[1][0]))
            nodes.append(item[0][0])
            nodes.append(item[0][1])
        nodes=list(set(nodes))
        G2=nx.Graph()
        G2.add_nodes_from(nodes)
        G2.add_weighted_edges_from(lst)


        for item in dict(G1.degree()).keys():
            teambornyear=int(teaminfo[teaminfo.TeamID==item].FirstD1Season)
            teamage=year-teambornyear
            edges_lastyear=G2.edges([item])
            rivals_lastyear=list(set([i[1] for i in edges_lastyear]))
            if dict_teamrivals.get(item,"Never")=="Never":
                dict_teamrivals[item]=rivals_lastyear
            else:
                s=dict_teamrivals[item]
                if rivals_lastyear is None:
                    s=s.extend(rivals_lastyear)
                    dict_teamrivals[item]=s
            edges_year=G1.edges([item])
            if type(dict_teamrivals[item]) is not None:
                num_rivals_new=len([j[1] for j in edges_year if j[1] not in dict_teamrivals[item]])
            else:
                num_rivals_new=len([j[1] for j in edges_year])
            ratio=num_rivals_new/(num_teams_year-len(rivals_lastyear)-1)
            results.append((item,teamage,ratio))
          
    return results

results=time_with_newedges(range(1986,2020))
    
ages=[]
probs=[]
for i in range(1,35):
    result_age=[c for a,b,c in results if b == i]
    prob=sum(result_age)/len(result_age)
    ages.append(i)
    probs.append(prob)
    
plt.figure(figsize=(10,8))
plt.plot(ages,probs,color='blue',label='True net')
plt.scatter(ages,probs,s=20,marker='s')
plt.title("Ages vs Newedges")
plt.xlabel("Team age")
plt.ylabel("Average probability of new edges")
plt.grid(c='black')
plt.show()
mdf = pd.read_csv(f"../input/export-dataframe/export_dataframe_MNCAA.csv")
wdf = pd.read_csv(f"../input/export-dataframe/export_dataframe_WNCAA.csv")
dfa = pd.read_csv(f"../input/export-dataframe/export_dataframe_MNCAAWNCAA.csv")
dfa_since2010 = dfa[dfa.Season>=2010]
fig, (axs0, axs1) = plt.subplots(ncols = 2, figsize = (18,6))
sns.boxplot(x = 'Season', y = 'FGA', hue='Sex', palette = ['b' , 'r'] , data = dfa_since2010,ax=axs0)
sns.boxplot(x = 'Season', y = 'FGP', hue='Sex', palette = ['b' , 'r'] , data = dfa_since2010,ax=axs1)
plt.show()
fig, (axs0, axs1) = plt.subplots(ncols = 2, figsize = (18,6))
sns.boxplot(x = 'Season', y = 'FGA3', hue='Sex', palette = ['b' , 'r'] , data = dfa_since2010, ax=axs0)
sns.boxplot(x = 'Season', y = 'FTA', hue='Sex', palette = ['b' , 'r'] , data = dfa_since2010, ax=axs1)
plt.show()
r1 = pd.DataFrame(mdf,columns=['FGA','FGM','FGP','FGA3','FGM3','FGP3','Score','Ast','OR','DR'])
corr = r1.corr(method='spearman')
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15,10))
plt.title("Concise MNCAA Correlation matrix")
sns.heatmap(corr, mask = mask, cmap= 'BuPu', annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

r2 = pd.DataFrame(wdf,columns=['FGA','FGM','FGP','FGA3','FGM3','FGP3','Score','Ast','OR','DR'])
corr = r2.corr(method='spearman')
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15,10)) 
plt.title("Concise WNCAA Correlation matrix")
sns.heatmap(corr, mask = mask, cmap= 'YlGnBu', annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
fig, (axs0,axs1) = plt.subplots(ncols = 2, figsize = (14,6))
sns.regplot(x='OR', y='FGA', data=mdf,ax=axs0,color='r').set(xlim = (0,60),ylim = (70,180),title = 'MNCAA FGA vs. OR')
sns.regplot(x='OR', y='FGA', data=wdf,ax=axs1,color='b').set(xlim = (0,60),ylim = (70,180),title = 'WNCAA FGA vs. OR')
plt.show()
fig, (axs0,axs1) = plt.subplots(ncols = 2, figsize = (14,6))
sns.regplot(x='Ast', y='FGP', data=mdf, ax=axs0,color='b').set(xlim = (5,55),ylim = (0.4,1.4),title = 'MNCAA FGP vs. Ast')
plt.grid()
sns.regplot(x='Ast', y='FGP', data=wdf, ax=axs1,color='r').set(xlim = (5,55),ylim = (0.4,1.4),title = 'WNCAA FGP vs. Ast')
plt.grid()
plt.show()
sns.set(style="ticks", palette="pastel")
ax = sns.violinplot(x="Area", y="FG%", hue="Sex",data=pd.read_csv(f"../input/boxplot/df_FG.csv"), palette="Set"+str(1),fliersize = 1)
ax.set_title("FG% in MEvents2019 & WEvents2019 in Different Area")
fig = ax.get_figure()
sns.set(style="ticks", palette="pastel")
ax = sns.violinplot(x="Area", y="FGA", hue="Sex",data=pd.read_csv(f"../input/boxplot/df_FGA.csv"), palette="Set"+str(1),fliersize = 1)
ax.set_title("FGA in MEvents2019 & WEvents2019 in Different Area")
fig = ax.get_figure()