#!pip install pandas-profiling --upgrade --user ##giving error   # To install pandas profiling please run this command.



#conda install -c conda-forge pandas-profiling                   #completed...uncomment if need to re-install



!pip install pivottablejs                                       #installing Pivot Table JS package



import numpy as np

np.set_printoptions(precision=4)                    # To display values only upto four decimal places. 



import pandas as pd

pd.set_option('mode.chained_assignment', None)      # To suppress pandas warnings.

pd.set_option('display.max_colwidth', -1)           # To display all the data in each column

pd.options.display.max_columns = 50                 # To display every column of the dataset in head()



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')                  # To apply seaborn whitegrid style to the plots.

%matplotlib inline



import seaborn as sns

sns.set(style='whitegrid', font_scale=1.3, color_codes=True)      # To adjust seaborn settings for the plots.



import warnings

warnings.filterwarnings('ignore')                   # To suppress all the warnings in the notebook.



import pandas_profiling                             # To get Pandas profiling reports  

from pivottablejs import pivot_ui                   # To get the Pivot table JS

import webbrowser                                   # To open the pivot table html file in web browser



import plotly.express as px                         # To import plotly package for interacrive graphs

                     



# Making bokeh specific imports.



from bokeh.plotting import Figure, figure, show, output_notebook,output_file

from bokeh.layouts import column

from bokeh.models import ColumnDataSource, CustomJS, Slider, HoverTool

from bokeh.palettes import Spectral4

from bokeh.io import push_notebook

from bokeh.models import Range1d

output_notebook()





from ipywidgets import interact                    # For bokeh slider and selections
pl_df = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Projects/English_Premier_League.csv')                               # read_csv is used to read csv file

pl_df.head()
pl_df.info()
pl_df.describe(include='all')
# To output pandas profiling report to an external html file.

# Saving the output as profiling_before_preprocessing.html

'''

profile = df_merge.profile_report(title='Pandas Profiling before Data Preprocessing')

profile.to_file(output_file="profiling_before_preprocessing.html")

'''



# To output the pandas profiling report on the notebook.



#pl_df.profile_report(title='Pandas Profiling before Data Preprocessing', style={'full_width':True})





#profile = pandas_profiling.ProfileReport(pl_df)

#profile.to_file(output_file="profiling_before_preprocessing.html")
pl_df.head()
pl_df.drop(['Div'],axis=1,inplace=True)
pl_df.head()
#pl_df['Date'] = pd.to_datetime(pl_df['Date'])

#pl_df.Date= pd.to_datetime(pl_df['Date'], format='%m/%d/%Y')  

## The above had issues as the original date had 2 formats and on further checking found that one date format got converted to mm/dd/yyyy and other to dd/mm/yyyy causing confusion.

pl_df['New_Date'] = pd.to_datetime(pl_df.Date , format ='%d/%m/%Y',errors='coerce')

fil = pl_df.New_Date.isnull()

pl_df.loc[fil,'New_Date'] = pd.to_datetime(pl_df[fil]['Date'],errors='coerce')

pl_df.info()

pl_df.head()
print(pl_df[pl_df['HTHG'].isnull()]['Season'].value_counts())

print(pl_df[pl_df['HTAG'].isnull()]['Season'].value_counts())

print(pl_df[pl_df['HTR'].isnull()]['Season'].value_counts())

print(pl_df['Season'].value_counts())
pl_df[['HTHG','HTAG']] = pl_df[['HTHG','HTAG']].fillna(value=0).astype(int)

pl_df['HTR'] = pl_df['HTR'].fillna(value='D')

pl_df.head()
#profile = pandas_profiling.ProfileReport(pl_df)

#profile.to_file(output_file="profiling_after_preprocessing.html")
pl_df.groupby('Season')['Season'].count().plot(kind='bar',fontsize=13,figsize=(16,7),yticks=np.arange(0,501,50))



plt.title("Matches Played per Season")

plt.xlabel("Season")

plt.ylabel("Matches")

plt.xticks(rotation=60)
#No. of Teams per season



pl_df.groupby('Season')['HomeTeam'].nunique().plot(kind='bar',colormap='magma',fontsize=13,figsize=(16,7),yticks=np.arange(0, 26))

plt.ylabel('No. of Home Teams')

plt.title("Total No. of Teams per Season")

plt.xticks(rotation=60)
pl_df['Total_Match_Goals'] = pl_df['FTHG'] + pl_df['FTAG']



pl_df.groupby('Season')['Total_Match_Goals'].sum().plot(kind='bar',fontsize=13,figsize=(16,7),yticks=np.arange(0,1201,80)) 



plt.title("Goals Scored per Season")

plt.xlabel("Season")

plt.ylabel("Goals")

plt.xticks(rotation=60)
gpm = round(pl_df.groupby('Season')['Total_Match_Goals'].sum()/pl_df.groupby('Season')['Season'].count(),2)

gpm.plot(kind='bar',width=0.5,figsize=(24,8))



plt.grid(axis='y')

plt.ylim(2.4,2.9)  ##min value = 2.45 as seen in data [gpm.min()]

plt.ylabel('Goals')

plt.title("Goals per Game",fontsize=20)

plt.xticks(rotation=60)
pl_df.groupby('Season')[['FTHG','FTAG']].sum().plot(kind='bar',fontsize=13,figsize=(24,8),yticks=np.arange(0,721,80),colormap='copper')

plt.grid(axis='y')

plt.ylabel('Goals')

plt.legend(['Home Goals','Away Goals'])

plt.title("Home Goals v/s Away Goals",fontsize=20)

plt.xticks(rotation=60)
round((pl_df.groupby('Season')['FTHG'].sum()/pl_df.groupby('Season')['FTAG'].sum()),3).plot(kind='bar',figsize=(24,8),color='orange')

plt.grid(axis='y')

plt.ylabel("Ratio")

plt.title("Home Goals/Away Goals",fontsize= 20)

plt.xticks(rotation=60)
pl_df['Comeback']=np.where((pl_df['FTR']!=pl_df['HTR'])&(pl_df['FTR']!='D') & (pl_df['HTR']!='D'),1,0)



pl_df.groupby('Season')['Comeback'].sum().plot(kind='bar',figsize=(24,8),color='green',yticks=np.arange(0,23,2))

plt.grid(axis='y')

plt.title("Comebacks per Season",fontsize=20)

plt.ylabel('Comebacks')

plt.xticks(rotation=60)
tot_g = pl_df.groupby('HomeTeam')['FTHG'].sum() + pl_df.groupby('AwayTeam')['FTAG'].sum()

tot_g.sort_values(ascending=False).plot(kind='barh',figsize=(15,15),xticks=np.arange(0,1901,100),color='darkorange')

plt.grid(axis='x')

plt.xlabel("Goals")

plt.title("Total Goals by Teams",fontsize=20)
pl_df.groupby('HomeTeam')['FTHG'].sum().sort_values(ascending=False).plot(kind='barh',figsize=(15,15),xticks=np.arange(0,1051,50),color='darkorange')

plt.grid(axis='x')

plt.xlabel("Goals")

plt.title("Home Goals by Teams",fontsize=20)

plt.legend(['Goals Scored'])
goals_h_game= round(pl_df.groupby('HomeTeam')['FTHG'].sum()/pl_df.groupby('HomeTeam')['HomeTeam'].count(),2)

goals_h_game.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,2.5,0.2),color='darkorange')

plt.grid(axis='x')

plt.xlabel("Goals")

plt.title("Home Goals/Game",fontsize=20)

plt.legend(['Goals Scored per Game'])
home_win=100*round((pl_df.loc[pl_df['FTR']=='H'].groupby('HomeTeam')['FTR'].count()/pl_df.groupby('HomeTeam')['FTR'].count()),3)

home_win.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,101,10),color='darkorange')

plt.grid(axis='x')

plt.xlabel('Percentage')

plt.legend(['% Wins'])

plt.title("Home Ground Win %",fontsize=20)
home_loss=100*round((pl_df.loc[pl_df['FTR']=='A'].groupby('HomeTeam')['FTR'].count()/pl_df.groupby('HomeTeam')['FTR'].count()),3)

home_loss.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,101,10),color='darkorange')

plt.grid(axis='x')

plt.xlabel('Percentage')

plt.legend(['% Loss'])

plt.title("Home Ground Loss %",fontsize=20)
home_draw=100*round((pl_df.loc[pl_df['FTR']=='D'].groupby('HomeTeam')['FTR'].count()/pl_df.groupby('HomeTeam')['FTR'].count()),3)

home_draw.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,101,10),color='darkorange')

plt.grid(axis='x')

plt.xlabel('Percentage')

plt.legend(['% Draw'])

plt.title("Home Ground Draw %",fontsize=20)
home_cb = pl_df.loc[(pl_df['HTR']=='A')].groupby('HomeTeam')['Comeback'].sum()

home_cb.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,21,2),color='darkorange')

plt.grid(axis='x')

plt.xlabel('Comeback')

plt.legend(['Comeback'])

plt.title("Home Comebacks",fontsize=20)
pl_df.groupby('AwayTeam')['FTAG'].sum().sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,851,50),color='teal')

plt.grid(axis='x')

plt.xlabel("Goals")

plt.title("Away Goals by Teams",fontsize=20)

plt.legend(['Goals Scored'])
goals_a_game= round(pl_df.groupby('AwayTeam')['FTAG'].sum()/pl_df.groupby('AwayTeam')['AwayTeam'].count(),2)

goals_a_game.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,2.5,0.2),color='teal')

plt.grid(axis='x')

plt.xlabel("Goals")

plt.title("Away Goals/Game",fontsize=20)

plt.legend(['Goals Scored per Game'])
away_win=100*round((pl_df.loc[pl_df['FTR']=='A'].groupby('AwayTeam')['FTR'].count()/pl_df.groupby('AwayTeam')['FTR'].count()),3)

away_win.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,101,10),color='teal')

plt.grid(axis='x')

plt.xlabel('Percentage')

plt.legend(['% Wins'])

plt.title("Away Ground Win %",fontsize=20)
away_win=100*round((pl_df.loc[pl_df['FTR']=='H'].groupby('AwayTeam')['FTR'].count()/pl_df.groupby('AwayTeam')['FTR'].count()),3)

away_win.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,101,10),color='teal')

plt.grid(axis='x')

plt.xlabel('Percentage')

plt.legend(['% Loss'])

plt.title("Away Ground Loss %",fontsize=20)
away_win=100*round((pl_df.loc[pl_df['FTR']=='D'].groupby('AwayTeam')['FTR'].count()/pl_df.groupby('AwayTeam')['FTR'].count()),3)

away_win.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,101,10),color='teal')

plt.grid(axis='x')

plt.xlabel('Percentage')

plt.legend(['% Draws'])

plt.title("Away Ground Draw %",fontsize=20)
away_cb = pl_df.loc[(pl_df['HTR']=='H')].groupby('AwayTeam')['Comeback'].sum()

away_cb.sort_values(ascending=False).plot(kind='barh',figsize=(10,15),xticks=np.arange(0,23,2),color='teal')

plt.grid(axis='x')

plt.xlabel('Comeback')

plt.legend(['Comeback'])

plt.title("Away Comebacks",fontsize=20)
(100*(home_cb/(home_cb+away_cb))).plot(kind='bar',figsize=(15,10),yticks=np.arange(0,101,5),color='teal')

plt.grid(axis='x')

plt.xlabel('Team')

plt.ylabel('Home Comeback %')

plt.legend(['Home Comeback %'])

plt.title("Home/Total Comebacks",fontsize=20)
pl_df.groupby(['HomeTeam'])['Season'].nunique().plot(kind='bar',figsize=(20,10),yticks=np.arange(0,26,1),color='black')

plt.ylabel("No. Of Seasons")

plt.xticks(rotation=90)

plt.title("Teams played in Different Seasons",fontsize=20)

plt.grid()
#Getting total Home Games, Wins, Draws and Losses

hp = pl_df.groupby(['Season','HomeTeam'])['FTR'].count().rename("HP")

hw = pl_df[pl_df['FTR'] =='H'].groupby(['Season','HomeTeam'])['FTR'].count().rename("HW")

hl = pl_df[pl_df['FTR'] =='A'].groupby(['Season','HomeTeam'])['FTR'].count().rename("HL")

hd = pl_df[pl_df['FTR'] =='D'].groupby(['Season','HomeTeam'])['FTR'].count().rename("HD")

home_games=pd.concat([hp,hw,hl,hd],axis=1).reset_index()

home_games.fillna(0,inplace=True)

home_games[['HP','HL','HD','HW']]=home_games[['HP','HL','HD','HW']].astype(int)





#Getting Total Goals scored and conceded by Home Teams per Season (Full Time and Half Time)

home_score=pl_df.groupby(['Season','HomeTeam'])['FTAG','FTHG','HTAG','HTHG'].sum().astype(int).reset_index()

home_score.rename(columns={'FTHG':'HG For','FTAG':'HG Against','HTAG':'HG For(Half Time)','HTHG':'HG Against(Half Time)'},inplace=True)



home_df=pd.merge(left=home_games,right=home_score,on=['Season','HomeTeam'],how='left')
#Getting total Away Games, Wins, Draws and Losses

ap = pl_df.groupby(['Season','AwayTeam'])['FTR'].count().rename("AP")

aw = pl_df[pl_df['FTR'] =='A'].groupby(['Season','AwayTeam'])['FTR'].count().rename("AW")

al = pl_df[pl_df['FTR'] =='H'].groupby(['Season','AwayTeam'])['FTR'].count().rename("AL")

ad = pl_df[pl_df['FTR'] =='D'].groupby(['Season','AwayTeam'])['FTR'].count().rename("AD")

away_games=pd.concat([ap,aw,al,ad],axis=1).reset_index()

away_games.fillna(0,inplace=True)   ##NaN comes where theere are 0 games as groupby misses that entry

away_games[['AP','AL','AD','AW']]=away_games[['AP','AL','AD','AW']].astype(int)





#Getting Total Goals scored and conceded by Away Teams per Season (Full Time and Half Time)

away_score=pl_df.groupby(['Season','AwayTeam'])['FTAG','FTHG','HTAG','HTHG'].sum().astype(int).reset_index()

away_score.rename(columns={'FTHG':'AG Against','FTAG':'AG For','HTAG':'AG For(Half Time)','HTHG':'AG Against(Half Time)'},inplace=True)



away_df=pd.merge(left=away_games,right=away_score,on=['Season','AwayTeam'],how='left')

##Final Points Table (Home and Away)



pt_table = pd.merge(left=home_df,right=away_df,left_on=['Season','HomeTeam'],right_on=['Season','AwayTeam']).drop(['AwayTeam'],axis=1)

pt_table.rename(columns={'HomeTeam':'Team'},inplace=True)

pt_table['Total Played'] = pt_table['HP'] + pt_table['AP']

pt_table['Total Wins'] = pt_table['HW'] + pt_table['AW']

pt_table['Total Loss'] = pt_table['HL'] + pt_table['AL']

pt_table['Total Draw'] = pt_table['HD'] + pt_table['AD']

pt_table['GF'] = pt_table['HG For'] + pt_table['AG For']

pt_table['GA'] = pt_table['HG Against'] + pt_table['AG Against']

pt_table['GD'] = pt_table['GF'] - pt_table['GA']

pt_table['Total_Points'] = 3*pt_table['Total Wins'] + pt_table['Total Draw']  ## 3 Points for a Win and 1 Point for a Draw.





## Ranking the teams

pt_table['Rank']= pt_table.groupby('Season')['Total_Points'].rank(method='min',ascending=False)

pt_table['fin'] = pt_table.groupby(['Season','Rank'])['GD'].rank(method='first',ascending=False)  ## If points are same, use goal difference

pt_table['Rank']=np.where((pt_table['fin'] > 1),(pt_table['Rank']+pt_table['fin'] - 1),pt_table['Rank'])

pt_table.drop('fin',axis=1,inplace=True)



#pt_table

#pt_table[pt_table['Season']=='2011-12'].sort_values(by=['Total_Points','GD'], ascending=[False,False])
pt_table[pt_table['Season']=='2011-12'].sort_values(by='Rank')[['Season','Team','Total Played','Total Wins','Total Loss','Total Draw','GF','GA','GD','Total_Points','Rank']]
fig,ax=plt.subplots(4,2,figsize=(40,40),constrained_layout=True)



## Most and Least Premier League wins (minimum 1)

pt_table[pt_table['Rank']==1].groupby('Team')['Season'].count().plot(kind='bar',ax=ax[0][0])

ax[0][0].set_title('League Wins ',fontsize=20)

ax[0][0].grid()





##Most and Least Wins in a season

pt_table.groupby('Season')['Total Wins'].agg(['max','min']).plot(kind='bar',ax=ax[0][1])

ax[0][1].set_title('Max and Min Wins ',fontsize=20)

ax[0][1].grid()





##Most and Least Points in a season

pt_table.groupby('Season')['Total_Points'].agg(['max','min']).plot(kind='bar',ax=ax[1][0])

ax[1][0].set_title('Max and Min Points ',fontsize=20)

ax[1][0].grid()



##Most and Least Defeats in a season

pt_table.groupby('Season')['Total Loss'].agg(['max','min']).plot(kind='bar',ax=ax[1][1])

ax[1][1].set_title('Max and Min Defeats ',fontsize=20)

ax[1][1].grid()



##Most and Least Goals in a season

pt_table.groupby('Season')['GF'].agg(['max','min']).plot(kind='bar',ax=ax[2][0])

ax[2][0].set_title('Max and Min Goals Scored ',fontsize=20)

ax[2][0].grid()



##Best and worst Goal Difference in a season

pt_table.groupby('Season')['GD'].agg(['max','min']).plot(kind='bar',ax=ax[2][1])

ax[2][1].set_title('Max and Min Goal Difference ',fontsize=20)

ax[2][1].grid()



##Most and Least Goals Conceded in a season

pt_table.groupby('Season')['GA'].agg(['max','min']).plot(kind='bar',ax=ax[3][0])

ax[3][0].set_title('Max and Min Goals Conceded ',fontsize=20)

ax[3][0].grid()



##Biggest and Smallest Margin between top 2 teams over Seasons



## Top Team Data for every Season to see trending

top = pt_table[(pt_table['Rank']==1)]

top.set_index('Season',drop=True,inplace=True)

top = top.reset_index() ##set and reset done becuase otherwise it was showing irregular spaces in x axis of the plot



## Runner Up Team Data for every Season to see trending

rup = pt_table[(pt_table['Rank']==2)]

rup.set_index('Season',drop=True,inplace=True)

rup = rup.reset_index() ##set and reset done becuase otherwise it was showing irregular spaces in x axis of the plot











top.set_index('Season',inplace=True)

rup.set_index('Season',inplace=True)



(top['Total_Points'] - rup['Total_Points']).plot(kind='bar',ax=ax[3][1],yticks=np.arange(0,21,1))

ax[3][1].set_title('Biggest and Smallest Margins for Top2 Teams ',fontsize=20)

ax[3][1].grid()



top.reset_index(inplace=True)

rup.reset_index(inplace=True)





#fig.delaxes(ax[3][1])

#fig.delaxes(ax[2][2])

fig.tight_layout()

fig.show()



fig = px.scatter(pt_table, x="Season", y="Total Loss", color="Team",hover_name="Team",size="Total Wins",template='plotly',

                color_discrete_sequence=px.colors.cyclical.IceFire,

                 #labels ={'GF': 'Goals Scored'},

                 title="Total Matches Lost per Season and  Most Wins reflected by the size")



fig.update_xaxes(

    showgrid=True,

    ticks="outside",

    tickson="boundaries",

    ticklen=20,

    type='category'

)





fig.show()
fig = px.scatter(pt_table, x="Season", y="GD", color="Team",hover_name="Team",size="Total_Points",template='plotly',

                color_discrete_sequence=px.colors.cyclical.IceFire,

                 labels ={'GD': 'Goal Difference'},

                 title="Goal Difference per Season with Total Points reflected by the size")



fig.update_xaxes(

    showgrid=True,

    ticks="outside",

    tickson="boundaries",

    ticklen=20,

    type='category'

)





fig.show()
fig = px.scatter(pt_table, x="Season", y="GF", color="Team",hover_name="Team",size="GA",template='plotly',

                color_discrete_sequence=px.colors.cyclical.IceFire,

                 labels ={'GF': 'Goals Scored'},

                 title="Goals Scored per Season with Goals Conceeded as size")



fig.update_xaxes(

    showgrid=True,

    ticks="outside",

    tickson="boundaries",

    ticklen=20,

    type='category'

)





fig.show()
ax = top.plot(kind='line',color='sandybrown',y='Total_Points',grid=True,figsize=(47,20),label='Total_Points',linestyle='-',marker='D',markevery=1,markersize=15, fillstyle='full',markerfacecolor='sandybrown')

plt.axhline(y=top['Total_Points'].mean(),color='red',label='Average Points',linewidth=4)

plt.yticks(np.arange(70,105,2),fontsize=20)

plt.annotate('Avg Points',(-1,top['Total_Points'].mean() + 0.5),fontsize = 40,fontweight='bold')

a=0

for index,row in top.iterrows():

    b=row.Total_Points

    plt.annotate(row.Team,(a,b+.5),fontsize=20,fontweight='bold')

    a=a+1

plt.xlabel('Season',fontsize=50)

plt.ylabel('Points',fontsize=50)

plt.title('Premier League Champions (Points over the years) ',fontsize=60)

plt.legend(loc=2,fontsize=35)

plt.xticks(top.index,rotation=60,fontsize=20)

ax.set_xticklabels(top['Season'])
ax = top.plot(kind='line',color='sandybrown',y='GF',grid=True,figsize=(47,20),label='Goals Scored',linestyle='-',marker='D',markevery=1,markersize=15, fillstyle='full',markerfacecolor='sandybrown')

plt.axhline(y=top['GF'].mean(),color='red',label='Average Goals Scored',linewidth=4)

plt.yticks(np.arange(65,111,3),fontsize=20)



plt.annotate('Avg Goals Scored',(-1,top['GF'].mean() + 0.5),fontsize = 40,fontweight='bold')

a=0

for index,row in top.iterrows():

    b=row.GF

    plt.annotate(row.Team,(a,b+.5),fontsize=20,fontweight='bold')

    a=a+1





plt.xlabel('Season',fontsize=50)

plt.ylabel('Goals',fontsize=50)

plt.title('Premier League Champions (Goals over the years) ',fontsize=60)

plt.legend(loc=2,fontsize=35)

plt.xticks(top.index,rotation=60,fontsize=20)

ax.set_xticklabels(top['Season'])
ax = rup.plot(kind='line',color='black',y='Total_Points',grid=True,figsize=(47,20),label='Total_Points',linestyle='-',marker='D',markevery=1,markersize=15, fillstyle='full',markerfacecolor='sandybrown')

plt.axhline(y=rup['Total_Points'].mean(),color='red',label='Average Points',linewidth=4)

plt.yticks(np.arange(60,93,2),fontsize=20)

plt.annotate('Avg Points',(-1,rup['Total_Points'].mean() + 0.5),fontsize = 40,fontweight='bold')

a=0

for index,row in rup.iterrows():

    b=row.Total_Points

    plt.annotate(row.Team,(a,b+.5),fontsize=20,fontweight='bold')

    a=a+1

plt.xlabel('Season',fontsize=50)

plt.ylabel('Points',fontsize=50)

plt.title('Premier League Runners Up (Points over the years) ',fontsize=60)

plt.legend(loc=4,fontsize=35)

plt.xticks(rup.index,rotation=60,fontsize=20)

ax.set_xticklabels(rup['Season'])
ax = rup.plot(kind='line',color='black',y='GF',grid=True,figsize=(47,20),label='Goals Scored',linestyle='-',marker='D',markevery=1,markersize=15, fillstyle='full',markerfacecolor='sandybrown')

plt.axhline(y=rup['GF'].mean(),color='red',label='Average Goals Scored',linewidth=4)

plt.yticks(np.arange(55,107,3),fontsize=20)



plt.annotate('Avg Goals Scored',(-1,rup['GF'].mean() + 0.5),fontsize = 40,fontweight='bold')

a=0

for index,row in rup.iterrows():

    b=row.GF

    plt.annotate(row.Team,(a,b+.5),fontsize=20,fontweight='bold')

    a=a+1





plt.xlabel('Season',fontsize=50)

plt.ylabel('Goals',fontsize=50)

plt.title('Premier League Runners Up (Goals over the years) ',fontsize=60)

plt.legend(loc=2,fontsize=35)

plt.xticks(rup.index,rotation=60,fontsize=20)

ax.set_xticklabels(rup['Season'])
## 18th postion team Data for every Season to see trending

rel = pt_table[(pt_table['Rank']==18) & (pt_table['Season'] != '1993-94') & (pt_table['Season'] != '1994-95')]

rel.set_index('Season',drop=True,inplace=True)

rel=rel.reset_index()   ##set and reset done becuase otherwise it was showing irregular spaces in x axis of the plot





ax = rel.plot(kind='line',color='sandybrown',y='Total_Points',figsize=(48,20),label='Total_Points',linestyle='-',marker='D',markevery=1,markersize=10, fillstyle='full',markerfacecolor='sandybrown')

plt.axhline(y=rel['Total_Points'].mean(),color='red',label='Average Points',linewidth=4)

plt.yticks(np.arange(25,50,2),fontsize=20)

plt.annotate('Avg Points',(-1,rel['Total_Points'].mean() + 0.5),fontsize = 40,fontweight='bold')

a=0

for index,row in rel.iterrows():

    b=row.Total_Points

    plt.annotate(row.Team,(a,b+.5),fontsize=20,fontweight='bold')

    a=a+1

plt.xlabel('Season',fontsize=50)

plt.ylabel('Points',fontsize=50)

plt.title('Premier League Relegation Border (Points over the years) ',fontsize=60)

plt.legend(loc=2,fontsize=35)

plt.xticks(rel.index,rotation=60,fontsize=20)

ax.set_xticklabels(rel['Season'])

plt.grid()
ax = rel.plot(kind='line',color='sandybrown',y='GF',grid=True,figsize=(47,20),label='Goals Scored',linestyle='-',marker='D',markevery=1,markersize=15, fillstyle='full',markerfacecolor='sandybrown')

plt.axhline(y=rel['GF'].mean(),color='red',label='Average Goals Scored',linewidth=4)

plt.yticks(np.arange(25,52,1),fontsize=20)



plt.annotate('Avg Goals Scored',(-1,rel['GF'].mean() + 0.5),fontsize = 40,fontweight='bold')

a=0

for index,row in rel.iterrows():

    b=row.GF

    plt.annotate(row.Team,(a,b+.5),fontsize=20,fontweight='bold')

    a=a+1





plt.xlabel('Season',fontsize=50)

plt.ylabel('Goals',fontsize=50)

plt.title('Premier League Relegation Border (Goals over the years) ',fontsize=60)

plt.legend(loc=2,fontsize=35)

plt.xticks(rel.index,rotation=60,fontsize=20)

ax.set_xticklabels(rel['Season'])
all_season = ['Arsenal','Chelsea','Liverpool','Everton','Man United','Tottenham']





pivot_ui(pt_table,rows=['Team'],cols=['Season'],vals=['Rank'],

         aggregatorName='List Unique Values',rendererName='Line Chart',

        inclusions= {'Team':all_season})

#Weekly Progress of Teams in the league by Game Week

## Skipping first 2 seasons as the rules changed from 1995 to allow only 20 teams (38 Game weeks)



league_prog = pl_df[(pl_df['Season'] !='1993-94') & (pl_df['Season'] !='1994-95')].copy()



def result(row): ##Function to assign HomeTeam or AwayTeam as a winner

    

    if(row['FTR']=='D'):

        return row['FTR']

    

    elif (row['Team'] == row['HomeTeam']):

        if (row['FTR'] == 'H'):

            return "W"

        else :

            return "L"

    elif (row['Team'] != row['HomeTeam']):

        if (row['FTR'] == 'A'):

            return "W"

        else:

            return "L"



def goaldiff(row): ##Function to assign HomeTeam or AwayTeam as a winner

    

    if (row['Team'] == row['HomeTeam']):

        return (row['FTHG']-row['FTAG'])

    else :

        return (row['FTAG']-row['FTHG'])

    

season_list=league_prog.Season.unique()

week_league=pd.DataFrame()

for j in season_list:

    team_list = league_prog[league_prog['Season']==j].HomeTeam.unique()

    season_temp=pd.DataFrame()

    for i in team_list:

        a = pd.DataFrame(league_prog[(league_prog['Season']==j) & ((league_prog['HomeTeam']==i)|(league_prog['AwayTeam']==i))].sort_values(by="New_Date")[['FTR','HomeTeam','Season','FTHG','FTAG']])

        a["Team"] = i

        a.loc[(a['Season']==j)&(a['Team']==i),'Game Week'] = np.arange(1,39)

        a['Game Week']=a['Game Week'].astype('Int64')

        a['Result']= a.apply(result,axis=1)

        a['GoalDiff']=a.apply(goaldiff,axis=1)

        a.drop(['FTR','HomeTeam'],axis=1,inplace=True)

        season_temp=pd.concat([season_temp,a])

    week_league=pd.concat([week_league,season_temp])



week_league['Points']=week_league['Result'].replace(['W','D','L'],[3,1,0]) ## Assign points as per Draw, Win or Loss

week_league['CumPoints']=week_league.groupby(['Season','Team'])['Points'].cumsum() ## Assign week over week Cumulative Points

week_league['TestRank']=week_league.groupby(['Season','Game Week'])['CumPoints'].rank(method='min',ascending=False)

week_league['fin'] = week_league.groupby(['Season','Game Week','TestRank'])['GoalDiff'].rank(method='first',ascending=False)

week_league['TestRank']=np.where((week_league['fin'] > 1),(week_league['TestRank']+week_league['fin'] - 1),week_league['TestRank'])

week_league.drop('fin',axis=1,inplace=True)

pivot_ui(week_league,rows=['Season','Team'],cols=['Game Week'],vals=['Result'],aggregatorName='List Unique Values',pvtTotal={'display':False},pvtTotalLabel={'display':False},pvtGrandTotal={'display':False},outfile_path='weekly_form.html')

pivot_ui(week_league,rows=['Season','Team'],cols=['Game Week'],vals=['CumPoints'],aggregatorName='Integer Sum',pvtTotal={'display':False},pvtTotalLabel={'display':False},pvtGrandTotal={'display':False},outfile_path='weekly_points.html')



new_1 = 2 # open in a new tab

url_1 = "weekly_form.html"

webbrowser.open(url_1,new=new_1)







new_2 = 2 # open in a new tab

url_2 = "weekly_points.html"

webbrowser.open(url_2,new=new_2)
#Update Plot Function to dynamically update the plot based on slider and Team values

def update_plot(season_no,team_no):

    season_list = list(week_league.Season.unique())

    team_list = list(week_league[week_league['Season']==season_list[season_no]].Team.unique())

    new_data = dict(x=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_list[team_no])]['Game Week'].tolist(), 

                  y=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_list[team_no])].CumPoints.tolist(), 

                  t=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_list[team_no])].Team.tolist(),

                  r=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_list[team_no])].Result.tolist(), 

                  s=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_list[team_no])].Season.tolist())

    source.data = new_data

    push_notebook()

    





# features to plot 

source = ColumnDataSource(dict(x=week_league[(week_league['Season']=='1995-96')&(week_league['Team']=='Arsenal')]['Game Week'].tolist(), 

                             y=week_league[(week_league['Season'] =='1995-96')&(week_league['Team']=='Arsenal')].CumPoints.tolist(), 

                             t=week_league[(week_league['Season'] =='1995-96')&(week_league['Team']=='Arsenal')].Team.tolist(), 

                             r=week_league[(week_league['Season'] =='1995-96')&(week_league['Team']=='Arsenal')].Result.tolist(), 

                             s=week_league[(week_league['Season']=='1995-96')&(week_league['Team']=='Arsenal')].Season.tolist()))



# define the plot size 

plot = Figure(plot_width=900, plot_height=700, tools=[HoverTool(tooltips=[('Week','@x'),('Points','@y'),('Result','@r')],show_arrow=False)], 

              x_axis_label='Game Week',y_axis_label='Points',title = 'League Progress Of a Team(Week over Week Change in Points)')



plot.circle(x='x', y='y', fill_alpha=10, source=source, color='blue',legend_field='s',size=5)

plot.line(x='x', y='y',legend_field='t',source=source,color='red',line_width=1.5)

plot.legend.location = 'top_left'

plot.legend.title = 'Season and Team'

plot.x_range = Range1d(start=1,end=38)

plot.y_range = Range1d(start=0,end=100)





from ipywidgets import interact

interact(update_plot, season_no=(0,22,1),team_no=(0,19,1))

show(plot, notebook_handle=True)
#Update Plot Function to dynamically update the plot based on slider values

def update_plot(season_no,team_no):

    season_list = list(week_league.Season.unique())

    #team_list = list(week_league[week_league['Season']==season_list[season_no]].Team.unique())

    new_data = dict(x=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_no)]['Game Week'].tolist(), 

                  y=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_no)].TestRank.tolist(), 

                  t=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_no)].Team.tolist(),

                  r=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_no)].Result.tolist(), 

                  s=week_league[(week_league['Season'] == season_list[season_no])&(week_league['Team']==team_no)].Season.tolist())

    source.data = new_data

    push_notebook()

    





# features to plot 

source = ColumnDataSource(dict(x=week_league[(week_league['Season']=='1995-96')&(week_league['Team']=='Arsenal')]['Game Week'].tolist(), 

                             y=week_league[(week_league['Season'] =='1995-96')&(week_league['Team']=='Arsenal')].TestRank.tolist(), 

                             t=week_league[(week_league['Season'] =='1995-96')&(week_league['Team']=='Arsenal')].Team.tolist(), 

                             r=week_league[(week_league['Season'] =='1995-96')&(week_league['Team']=='Arsenal')].Result.tolist(), 

                             s=week_league[(week_league['Season']=='1995-96')&(week_league['Team']=='Arsenal')].Season.tolist()))



# define the plot size 

plot = Figure(plot_width=900, plot_height=700, tools=[HoverTool(tooltips=[('Week','@x'),('Rank','@y'),('Team','@t')],show_arrow=False)], 

              x_axis_label='Game Week',y_axis_label='Rank',title = 'League Progress Of a Team(Change in Ranks by Game Week)')



plot.circle(x='x', y='y', fill_alpha=10, source=source, color='blue',legend_field='s',size=5)

plot.line(x='x', y='y',legend_field='t',source=source,color='red',line_width=1.5)

plot.legend.location = 'top_center'

plot.legend.title = 'Season and Team'

plot.x_range = Range1d(start=1,end=38)

plot.y_range = Range1d(start=0,end=21)





from ipywidgets import interact

interact(update_plot, season_no=(0,22,1),team_no=list(week_league.Team.unique()))

show(plot, notebook_handle=True)
sns.pairplot(data=pt_table[['GF', 'GA', 'GD','Rank']], size=2.5, diag_kind='kde')

scored=pt_table.groupby('Season')['GF'].max().rename("Max Goals Scored")

concd=pt_table.groupby('Season')['GA'].min().rename("Min Goals Conceded")

diff=pt_table.groupby('Season')['GD'].max().rename("Max Goals Difference")

max_min=pd.concat([scored,concd,diff],axis=1).reset_index()

max_min



top["Stat1"]=np.where((top['Season']==max_min['Season'])&(top['GF']==max_min['Max Goals Scored']),1,0)

top["Stat2"]=np.where((top['Season']==max_min['Season'])&(top['GA']==max_min['Min Goals Conceded']),1,0)

top["Stat3"]=np.where((top['Season']==max_min['Season'])&(top['GD']==max_min['Max Goals Difference']),1,0)

Goal_Scored= 100*top[top['Stat1']==1]['Stat1'].count()/top['Stat1'].count()

Goal_Conceded=100*top[top['Stat2']==1]['Stat2'].count()/top['Stat2'].count()

Goal_Difference= 100*top[top['Stat3']==1]['Stat3'].count()/top['Stat3'].count()

All3=100*(top[(top['Stat1']==1)&(top['Stat2']==1)&(top['Stat3']==1)]['Stat1'].count()/top['Stat3'].count())

NotAny=100*(top[(top['Stat1']!=1)&(top['Stat2']!=1)&(top['Stat3']!=1)]['Stat1'].count()/top['Stat3'].count())



my_list=[Goal_Scored,Goal_Conceded,Goal_Difference,All3,NotAny]

my_list1=['Most Scored','Least Conceded','Best Goal Difference','Best in All 3','Not in Any Tops']

Win=pd.Series(my_list, index=my_list1)



##setting Font

font = {'family': 'fantasy',

        'color':  'navy',

        'weight': 'normal',

        'size': 30,

        }



#Plot

Win.plot(kind='barh',figsize=(20,10),color=['red','blue','green','black','sandybrown'])   #colormap='Pastel2'

plt.xlabel("Percentage",fontsize=20)

plt.title('Percentage of Titles Won',fontdict=font)

plt.xticks(np.arange(0,101,5),fontsize=15)

plt.yticks(fontsize=15)

plt.grid(True)



#Annotate

a=0

for i in Win:

    plt.text(i-4,a-0.05 ,s=str(int(i)) +'%', horizontalalignment= 'center', verticalalignment='bottom', fontdict={'family':'serif','size':20,'color':'white','weight':'bold'})

    a=a+1








