# Importing necessary libraries needed for the analysis

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.

# read the input files and look at the top few observations in the dataset

data_path = "../input/"
epl_results = pd.read_csv(data_path+"EPL_Set.csv", dtype='unicode', encoding="utf-8-sig")
epl_results.head()
epl_results.head(20)
# Changing datatypes

epl_results["Season"] = epl_results["Season"].astype('str')
epl_results["HomeTeam"] = epl_results["HomeTeam"].astype('category')
epl_results["AwayTeam"] = epl_results["AwayTeam"].astype('category')
epl_results["FTHG"] = epl_results["FTHG"].astype(int)
epl_results["FTAG"] = epl_results["FTAG"].astype(int)
epl_results["FTR"] = epl_results["FTR"].astype('category')
epl_results["HTHG"] = epl_results["HTHG"].astype(float)
epl_results["HTAG"] = epl_results["HTAG"].astype(float)
epl_results["HTR"] = epl_results["HTR"].astype('category')
epl_results.dtypes
epl_results["HomeTeam"].value_counts(dropna=False)
# Number of home games played by each team from 1993 till May 2018

plt.figure(figsize=(17, 6))
sns.countplot(x='HomeTeam', data=epl_results)
plt.xticks(rotation='vertical')
plt.show()
# Number of Away games played by each team from 1993 till May 2018

plt.figure(figsize=(17, 6))
sns.countplot(x='AwayTeam', data=epl_results)
plt.xticks(rotation='vertical')
plt.show()
plt.rcParams['figure.figsize'] = [18, 15]
epl_results[['HomeTeam',
 'FTHG']].groupby('HomeTeam').sum().plot.barh(
 color='orange')
plt.title('')
plt.rcParams['figure.figsize'] = [18, 15]
epl_results[['AwayTeam',
 'FTAG']].groupby('AwayTeam').sum().plot.barh(
 color='blue')
plt.title('')
utd = epl_results.groupby(['Season','HomeTeam']).FTHG.sum().reset_index()
utd = utd.loc[utd['HomeTeam'] == 'Man United']
plt.rcParams['figure.figsize'] = [20, 18]
sns.barplot(x='Season', y='FTHG', data=utd)
autd = epl_results.groupby(['Season','AwayTeam']).FTAG.sum().reset_index()
autd = autd.loc[autd['AwayTeam'] == 'Man United']
plt.rcParams['figure.figsize'] = [20, 20]
sns.barplot(x='Season', y='FTAG', data=autd)
ars = epl_results.groupby(['Season','HomeTeam']).FTHG.sum().reset_index()
ars = ars.loc[ars['HomeTeam'] == 'Arsenal']
#plt.rcParams['figure.figsize'] = [20, 20]
#sns.barplot(x='Season', y='FTHG', data=ars)
ars.sort_values(['FTHG'], ascending=False)
import seaborn as sns

plt.hlines(y=my_range, xmin=0, xmax=ordered_df['FTHG'], color='skyblue')
plt.plot(ordered_df['FTHG'], my_range, "D")
plt.yticks(my_range, ordered_df['Season'])

plt.rcParams['figure.figsize'] = [10, 10]

# Reorder it following the values:
ordered_df = ars.sort_values(by='FTHG')
my_range=range(1,len(ars.index)+1)
 
# Make the plot
plt.hlines(y=my_range, xmin=0, xmax=ordered_df['FTHG'], color='skyblue')
plt.plot(ordered_df['FTHG'], my_range, "D")
plt.yticks(my_range, ordered_df['Season'])
plt.show()
aars = epl_results.groupby(['Season','AwayTeam']).FTAG.sum().reset_index()
aars = aars.loc[aars['AwayTeam'] == 'Arsenal']
plt.rcParams['figure.figsize'] = [20, 20]
sns.barplot(x='Season', y='FTAG', data=aars)
conditions = [epl_results['FTR']=='A',epl_results['FTR']=='H',epl_results['FTR']=='D']
conditions
select = [epl_results['AwayTeam'],epl_results['HomeTeam'],'Draw']
epl_results['Winner']=np.select(conditions, select)
swin = epl_results.loc[:,['Season','Winner']]
gswin = swin.groupby(['Season', 'Winner']).size().reset_index(name='counts')
gsort = gswin.sort_values(['Season', 'counts'], ascending=[True, False])
gsort
# Removing the draws out of the dataframe

rem_draws = gsort[gsort.Winner.str.contains('Draw')==False].reset_index()
rem_draws

#Now we can print the Team that has won most of the matches in each Season.

most_wins= rem_draws.groupby('Season').head(1)
most_wins
plt.figure(figsize=(17, 6))
sns.countplot(x='Winner', data=most_wins)
titles= most_wins['Winner'].value_counts().reset_index()
titles
plt.figure(figsize=(15, 10))
sns.barplot(x='index', y='Winner', data=titles)
manutd = epl_results.loc[epl_results['Winner'] == 'Man United']
count_series = manutd.groupby(['Season']).size()
mancount = pd.DataFrame(data=count_series, columns=['Wins']).reset_index()
mancount
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(20, 20))

plt.plot('Season', 'Wins', data=mancount, marker='o', color='skyblue')

plt.show()
# removing observations with blank half-time scores

epl_half = epl_results.dropna()
epl_half.head()
# Lets label Half-time winners, similar to the way we did for full-time winners

ht_win =[epl_half['HTR']=='A', epl_half['HTR']=='H',epl_half['HTR']=='D']
sel1 = [epl_half['AwayTeam'], epl_half['HomeTeam'],'Draw']
epl_half['Half_Time_Winner'] = np.select(ht_win, sel1)
# tabulating half-time winners

ht_df = epl_half.loc[:,['Season','Half_Time_Winner']]
gp_ht = ht_df.groupby(['Season', 'Half_Time_Winner']).size().reset_index(name='counts')
sort_ht= gp_ht.sort_values(['Season', 'counts'], ascending=[True, False])
sort_ht
rem_draw_ht = sort_ht[sort_ht.Half_Time_Winner.str.contains('Draw')==False]
rem_draw_ht
most_ht = rem_draw_ht.groupby('Season').head(1)
most_ht
plt.figure(figsize=(15, 10))
sns.countplot(x='Half_Time_Winner', data=most_ht)
epl_half
epl_half['Same Winner']=np.where(epl_half['Winner']==epl_half['Half_Time_Winner'],1,0)
epl_half
# Teams that have won both at half-time & at full-time, excluding draws

rem_draws = epl_half[(epl_half.Winner.str.contains('Draw')==False)&(epl_half.Half_Time_Winner.str.contains('Draw')==False)]

epl_half[epl_half['Same Winner'] ==1].head()

x = (int((rem_draws['Same Winner'].sum())) / int(len(rem_draws)))
print(x * 100)
epl_half_nodraw = epl_half[epl_half.Half_Time_Winner.str.contains('Draw')==False]
epl_half_nodraw = epl_half_nodraw[epl_half_nodraw.Winner.str.contains('Draw')==False]
epl_half_nodraw['Same Winner']=np.where(epl_half_nodraw['Winner']==epl_half_nodraw['Half_Time_Winner'],1,0)
epl_half['Same Winner']=np.where(epl_half['Winner']==epl_half['Half_Time_Winner'],1,0)
#Now we define a datset for no draw data with the full dataset
epl_results_nodraw = epl_results[epl_results.Winner.str.contains('Draw')==False]
# # Games that don't end in a draw if its a draw at half-time 

epl_half[(epl_half['Half_Time_Winner']=='Draw') & (epl_half['Same Winner']==0)]
# Games that end in a draw if its a draw at half-time

epl_half[(epl_half['Winner']=='Draw') & (epl_half['Same Winner']==1)]

# ~ 1400 odd games 
#Let us now see the number of goals per season.

epl_results['Total Goals']= epl_results['FTHG'] + epl_results['FTAG']
loca = epl_results.loc[:,['Season','Total Goals']]
summ =loca.groupby('Season').sum().reset_index()
tot = summ['Total Goals']
plt.figure(figsize=(15,8))
grid=sns.barplot(x='Season',y='Total Goals',data=summ,color='brown')
grid.set_xticklabels(summ['Season'],rotation=90)
plt.axhline(tot.mean())
plt.show()
#To check for Home Advantage exluding draws

len(epl_results[epl_results['FTR']=='H']) / len(epl_results_nodraw) * 100
rem_draws1 = gsort[gsort.Winner.str.contains('Draw')==False].reset_index()
rem_draws1

top4 = rem_draws1.groupby('Season').head(4)
valc = top4['Winner'].value_counts().reset_index()
plt.figure(figsize=(15,8))
grid1=sns.barplot(x='index',y='Winner',data= valc, color='violet')
grid1.set_xticklabels(valc['index'],rotation=90)
plt.show()
# Man United and Chelsea

manche = epl_results[((epl_results['HomeTeam']=='Man United' )& (epl_results['AwayTeam']=='Chelsea'))|((epl_results['HomeTeam']=='Chelsea') & (epl_results['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(manche.index))
sizes = (np.array((manche / manche.sum())*100))
colors = ['gold', 'lightskyblue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Man United and Chelsea")
plt.show()
                # Man United and Arsenal

manars = epl_results[((epl_results['HomeTeam']=='Man United' )& (epl_results['AwayTeam']=='Arsenal'))|((epl_results['HomeTeam']=='Arsenal') & (epl_results['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(manars.index))
sizes = (np.array((manars / manars.sum())*100))
colors = ['Pink', 'Violet']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Man United and Arsenal")
plt.show()
                # Man United and Liverpool

manliv = epl_results[((epl_results['HomeTeam']=='Man United' )& (epl_results['AwayTeam']=='Liverpool'))|((epl_results['HomeTeam']=='Liverpool') & (epl_results['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(manliv.index))
sizes = (np.array((manliv / manliv.sum())*100))
colors = ['Red', 'Lightblue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Man United and Liverpool")
plt.show()
                # Man United and Man City

mancity = epl_results[((epl_results['HomeTeam']=='Man United' )& (epl_results['AwayTeam']=='Man City'))|((epl_results['HomeTeam']=='Man City') & (epl_results['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(mancity.index))
sizes = (np.array((mancity / mancity.sum())*100))
colors = ['Red', 'blue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Man United and City")
plt.show()
                # Chelsea and Arsenal

chears = epl_results[((epl_results['HomeTeam']=='Chelsea' )& (epl_results['AwayTeam']=='Arsenal'))|((epl_results['HomeTeam']=='Arsenal') & (epl_results['AwayTeam']=='Chelsea'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(chears.index))
sizes = (np.array((chears / chears.sum())*100))
colors = ['Red', 'blue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Chelsea and Arsenal")
plt.show()
                # Chelsea and Arsenal

cheliv = epl_results[((epl_results['HomeTeam']=='Chelsea' )& (epl_results['AwayTeam']=='Liverpool'))|((epl_results['HomeTeam']=='Liverpool') & (epl_results['AwayTeam']=='Chelsea'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(cheliv.index))
sizes = (np.array((cheliv / cheliv.sum())*100))
colors = ['Red', 'blue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Chelsea and Liverpool")
plt.show()
                # Chelsea and Man City

chemanc = epl_results[((epl_results['HomeTeam']=='Chelsea' )& (epl_results['AwayTeam']=='Man City'))|((epl_results['HomeTeam']=='Man City') & (epl_results['AwayTeam']=='Chelsea'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(chemanc.index))
sizes = (np.array((chemanc / chemanc.sum())*100))
colors = ['Red', 'blue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Chelsea and Man City")
plt.show()
                # Liverpool and Arsenal

livars = epl_results[((epl_results['HomeTeam']=='Liverpool' )& (epl_results['AwayTeam']=='Arsenal'))|((epl_results['HomeTeam']=='Arsenal') & (epl_results['AwayTeam']=='Liverpool'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(livars.index))
sizes = (np.array((livars / livars.sum())*100))
colors = ['Red', 'blue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Liverpool and Arsenal")
plt.show()
                # Liverpool and Man City

livmanc = epl_results[((epl_results['HomeTeam']=='Liverpool' )& (epl_results['AwayTeam']=='Man City'))|((epl_results['HomeTeam']=='Man City') & (epl_results['AwayTeam']=='Liverpool'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(livmanc.index))
sizes = (np.array((livmanc / livmanc.sum())*100))
colors = ['Red', 'blue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Liverpool and Man City")
plt.show()
                # Arsenal and Man City

arsmanc = epl_results[((epl_results['HomeTeam']=='Arsenal' )& (epl_results['AwayTeam']=='Man City'))|((epl_results['HomeTeam']=='Man City') & (epl_results['AwayTeam']=='Arsenal'))].groupby(['Winner'])['Winner'].count()
labels = (np.array(arsmanc.index))
sizes = (np.array((arsmanc / arsmanc.sum())*100))
colors = ['Red', 'blue']
plt.subplots(figsize=(10, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win % Between Arsenal and Man City")
plt.show()
#Crosstabulation between the season and number of matches won by HomeTeam, AwayTeam and Season

df = pd.crosstab(index = epl_results["Season"],  columns=epl_results["FTR"],rownames=['Season']) 
df = pd.DataFrame(df);
#Plotting the above cross tabulation

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, ranges, LabelSet
from bokeh.plotting import figure
from bokeh.models import BoxSelectTool
from bokeh.io import push_notebook, output_notebook, curdoc
output_notebook()

source = ColumnDataSource(data=dict(df))

x = np.arange(25)\

p = figure(plot_width=1000, plot_height=600,title="Total number of Winners in Away, Home and Draw Games Each Season",
          y_range=epl_results['Season'].unique())

p.hbar(y=np.arange(1,26)-0.7, height=0.2, left=0,
       right=df['A'], color="violet",legend="Away Team")
p.hbar(y=np.arange(1,26)-0.5, height=0.2, left=0,
       right=df['H'], color="green",legend="Home Team")
p.hbar(y=np.arange(1,26)-0.3, height=0.2, left=0,
       right=df['D'], color="blue",legend="Draw")
plot = figure(tools="pan,wheel_zoom,box_zoom,reset")
plot.add_tools(BoxSelectTool(dimensions="width"))

show(p)
