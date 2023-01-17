import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Importing the EPL dataset.
epl_data = pd.read_csv('../input/EPL_Set.csv')
#Printing the head of the dataset.
epl_data.head()
#We can see that half time data is missing for the first two years.
#We can subset our data according to our need.
#Creating a column to store the winners of full time. 
conditions=[epl_data['FTR']=='A',epl_data['FTR']=='H',epl_data['FTR']=='D']
choices = [epl_data['AwayTeam'],epl_data['HomeTeam'],'Draw']
epl_data['Winner']=np.select(conditions,choices)
#We tabulate the data so that for every season we can see how many matches each Team wins.
a=epl_data.loc[:,['Season','Winner']]
b=a.groupby(['Season', 'Winner']).size().reset_index(name='counts')
c=b.sort_values(['Season', 'counts'], ascending=[True, False])
c
#Thus from the above data we can see that every season has a lot of draws. Though these matches add points to the 
#overall score for each team, it cannot be taken as a match won and thus to further analyse we can check without the
#Draw matches.
d=c[c.Winner.str.contains('Draw')==False].reset_index()
d
#Now we can print the Team that has won most of the matches in each Season.
e=d.groupby('Season').head(1)
e
#Now that we know which Team has won the most number of matches every year. We can check which team has won the most
#for the past 26 years.
z=e['Winner'].value_counts().reset_index()
z
#Clearly Man United has perfroming well for the past 26 years. It has the highest number of wins 11 out of 26 years.
#There is a huge gap between the good performers in 1st and 2nd place. Chelsea thus has won 5 Seasons in the past 
#26 years.
#However since this data is bound by time it cannot be said that it is true for all of its historical data.
#Plotting the Teams and number of Seasons they have won in the past 26 years.
sns.barplot(x='index', y='Winner', data=z, palette='Reds')
#Man United, Chelsea, Arsenal and Man City are the top 4 contenders.

#Now that we have analysed for team performance at the end of full time, we can now check for the performance at 
#half-time.
#As we do not have information for the first two Seasons, we shall subset the data for which half-time data is 
#available.
epl_data_half=epl_data.dropna()
epl_data_half.head()
#Now that we have subset our data, we label the winners for half_time
conditions1=[epl_data_half['HTR']=='A',epl_data_half['HTR']=='H',epl_data_half['HTR']=='D']
choices1 = [epl_data_half['AwayTeam'],epl_data_half['HomeTeam'],'Draw']
epl_data_half['Half_Time_Winner']=np.select(conditions1,choices1)
#Now let us tabulate data so that we can how many times each team wins in every Season.
#We also order the data for every year. 
f=epl_data_half.loc[:,['Season','Half_Time_Winner']]
g=f.groupby(['Season', 'Half_Time_Winner']).size().reset_index(name='counts')
h=g.sort_values(['Season', 'counts'], ascending=[True, False])
h
#There are a large number of draws even at half time.
#We can check the games won by ignoring the games that are at a draw during half time. 
i=h[h.Half_Time_Winner.str.contains('Draw')==False]
i
#Now that we have how many games each team wins every season we can check which team wins the most games in each
#season at half time. 
j=i.groupby('Season').head(1)
j
#Thus overall at half time throughout all the seasons we can see what teams are wining how mnay times.
y=j['Half_Time_Winner'].value_counts().reset_index()
y
#There are a lot more teams here than at full time, this shows that not all teams even though they seem to be winning
#in the first half, they do not win the match.
#Here the number of times the top teams have won decreases and some other teams are winning instead.
#This shows that the top teams are able to win even when they are not winning in the first half, however most of the 
#bottom teams are not able to do this.
#This also means there are a lot of draws during the first half.
#This plots the teams that win during half-time.
sns.barplot(x='index', y='Half_Time_Winner', data=y, palette='Blues')
#Now that we data on which teams win during the first half and which teams win at the end, we can comapare them for
#some insights.
#We compare them by checking all the games.
#To check for the teams winning both at half time and at the end, we use our subsetted data.
epl_data_half.head()
#Now we can check to see if the same team wins both at half time and at the end.
k=epl_data_half
k['Same Winner']=np.where(k['Winner']==k['Half_Time_Winner'],1,0)
k
#We can see the teams that have won at both isntances. However this includes draw.
k[k['Same Winner']==1].head()
x=k[(k.Winner.str.contains('Draw')==False)&(k.Half_Time_Winner.str.contains('Draw')==False)]
print(x['Same Winner'].sum())
print(len(x))
#About 3868 times out of 7324 matches the team who wins at half time also wins at full time

print(3864/4207)
#Thus we can say 92% of the time the match is won by the team wining at half time. this is true iff a draw match is not
#considered.

q=k[(k.Winner.str.contains('Draw')==False)|(k.Half_Time_Winner.str.contains('Draw')==False)]
q.head()
print(len(q))
print(x['Same Winner'].sum())
print(3868/7324)
#If draws are included then there is a 52% chance the match is won by the team at half time or ends in a draw
epl_data_half.head()
#We define a new dataset that contains all the data that has no Draw data.
epl_data_half_nodraw=epl_data_half[epl_data_half.Half_Time_Winner.str.contains('Draw')==False]
epl_data_half_nodraw=epl_data_half_nodraw[epl_data_half_nodraw.Winner.str.contains('Draw')==False]
epl_data_half_nodraw['Same Winner']=np.where(epl_data_half_nodraw['Winner']==epl_data_half_nodraw['Half_Time_Winner'],1,0)
epl_data_half['Same Winner']=np.where(epl_data_half['Winner']==epl_data_half['Half_Time_Winner'],1,0)
#Now we define a datset for no draw data with the full dataset
epl_data_nodraw=epl_data[epl_data.Winner.str.contains('Draw')==False]
#This gives us the games that are draw at half time but are won by a team in the end.
#There are 2236 such games.
epl_data_half[(epl_data_half['Half_Time_Winner']=='Draw') & (epl_data_half['Same Winner']==0)]
#These are the games that end in draw when the game is at draw at half time also.
#There are 1416 such games
epl_data_half[(epl_data_half['Winner']=='Draw') & (epl_data_half['Same Winner']==1)]
#There we can say the 62% of the games that are at draw at half time will be won by a team. 
print(2263/(1416+2263))

#If we include draw matches and check whether that the result at half time is the result at the end.
#There is a 66% chance that it will remain the same.
print(5824/8740)

#Removing matches that are draw at both times we get a 60% chance that we get the same result. 
print((5824-1416)/(8740-1416))

#We can see that the success rate falls when draw matches are included.
#Let us now see that number of goals per season.
epl_data['Total_Goals']=epl_data['FTHG']+epl_data['FTAG']
l=epl_data.loc[:,['Season','Total_Goals']]
m=l.groupby('Season').sum().reset_index()
y=m['Total_Goals']
plt.figure(figsize=(15,8))
grid=sns.barplot(x='Season',y='Total_Goals',data=m,color='orange')
grid.set_xticklabels(m['Season'],rotation=90)
plt.axhline(y.mean())
plt.show()
#The seasons of 93-94 and 94-95 have the highest number of goals.
#Seasons between 10 and 14 have done better than average number of goals. 
#However no years since 95 have been better for the number of goals.
#To check for Home Advantage.
epl_data[epl_data['FTR']=='H']
#These are all the times a Home team has won.

#we check the number matches that are not draw 
print(len(epl_data_nodraw))
print(4461/7118)
#Thus the home team wins 62% of the time. Thus there is definitely an advantage when playing on the home field,
#however the difference is not too large.
#Now let us see which teams take the first 4 spots in the every season.
n=d.groupby('Season').head(4)
o=n['Winner'].value_counts().reset_index()
plt.figure(figsize=(15,8))
grid1=sns.barplot(x='index',y='Winner',data=o,color='dodgerblue')
grid1.set_xticklabels(o['index'],rotation=90)
plt.show()
#Liverpool performs well in the first half and makes it to the top 4 however it is not able to win the season+-
#Let us compare the top 5 teams with eacb other, to check and see how matches are won by which team and how many
#end in a draw.
#Comparing the games between Man U and Chelsea 
epl_data[((epl_data['HomeTeam']=='Man United' )& (epl_data['AwayTeam']=='Chelsea'))|((epl_data['HomeTeam']=='Chelsea') & (epl_data['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
#We can see that even though Chelsea wins against Man U more than Man U does, it does not the overall season.
#Comparing the games between Man U and Arsenal. 
epl_data[((epl_data['HomeTeam']=='Man United' )& (epl_data['AwayTeam']=='Arsenal'))|((epl_data['HomeTeam']=='Arsenal') & (epl_data['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
#Man U wins most of the time.
#Comparing the games between Man U and Liverpool.
epl_data[((epl_data['HomeTeam']=='Man United' )& (epl_data['AwayTeam']=='Liverpool'))|((epl_data['HomeTeam']=='Liverpool') & (epl_data['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
#Man U wins more than Liverpool does
#Comparing the games between Man U and Man City.
epl_data[((epl_data['HomeTeam']=='Man United' )& (epl_data['AwayTeam']=='Man City'))|((epl_data['HomeTeam']=='Man City') & (epl_data['AwayTeam']=='Man United'))].groupby(['Winner'])['Winner'].count()
#Man U wins against Man City most of the time.
#Thus Man U is rightly on top of the charts with tough competetion faced only with Chelsea.
#Comparing the games between Arsenal and Chelsea.
epl_data[((epl_data['HomeTeam']=='Arsenal' )& (epl_data['AwayTeam']=='Chelsea'))|((epl_data['HomeTeam']=='Chelsea') & (epl_data['AwayTeam']=='Arsenal'))].groupby(['Winner'])['Winner'].count()
#Even though Arsenal wins against Chelsea more times, Chelsea has won the overall season more than Arsenal.
#Comparing the games between Arsenal and Liverpool.
epl_data[((epl_data['HomeTeam']=='Arsenal' )& (epl_data['AwayTeam']=='Liverpool'))|((epl_data['HomeTeam']=='Liverpool') & (epl_data['AwayTeam']=='Arsenal'))].groupby(['Winner'])['Winner'].count()
#Even though Liverpool has won more against Arsenal, they have not been able to win a season in the past 26 years.
#Comparing the games between Arsenal and Man City.
epl_data[((epl_data['HomeTeam']=='Arsenal' )& (epl_data['AwayTeam']=='Man City'))|((epl_data['HomeTeam']=='Man City') & (epl_data['AwayTeam']=='Arsenal'))].groupby(['Winner'])['Winner'].count()
#Arsenal is a better team than Man City.
#Comparing the games between Chelsea and Liverpool.
epl_data[((epl_data['HomeTeam']=='Chelsea' )& (epl_data['AwayTeam']=='Liverpool'))|((epl_data['HomeTeam']=='Liverpool') & (epl_data['AwayTeam']=='Chelsea'))].groupby(['Winner'])['Winner'].count()
#There are both good teams but Chelsea manages to win the season however liverpool has not.
#Comparing the games between Chelsea and Man City.
epl_data[((epl_data['HomeTeam']=='Chelsea' )& (epl_data['AwayTeam']=='Man City'))|((epl_data['HomeTeam']=='Man City') & (epl_data['AwayTeam']=='Chelsea'))].groupby(['Winner'])['Winner'].count()
#Chelsea is a better team than Man City.
#Comparing the games between Chelsea and Liverpool.
epl_data[((epl_data['HomeTeam']=='Man City' )& (epl_data['AwayTeam']=='Liverpool'))|((epl_data['HomeTeam']=='Liverpool') & (epl_data['AwayTeam']=='Man City'))].groupby(['Winner'])['Winner'].count()
#Even though Liverpool wins most of the time against Man City, Man City has won the overall seasons while Liverpool
#has not in the 26 years.
#Extracting the day, month and year for each EPL match
epl_data['day']=epl_data['Date'].str.split("/").str[0]
epl_data['month']=epl_data['Date'].str.split("/").str[1]
epl_data['year']=epl_data['Date'].str.split("/").str[2]
#Converting the datatype to integer format
epl_data['year']=epl_data['year'].astype(int)
epl_data['month']=epl_data['month'].astype(int)
epl_data['day']=epl_data['day'].astype(int)
#Keeping the year format as 'YYYY'
di = {93:1993,94:1994,95:1995,96:1996,97:1997,98:1998,99:1999,0:2000,1:2001,2:2002,3:2003,4:2004,5:2005,6:2006,
7:2007,8:2008,9:2009,10:2010,11:2011,12:2012,13:2013,14:2014,15:2015,16:2016,17:2017,18:2018}
epl_data=epl_data.replace({"year": di})
#Finding the winning team of each match and storing it in 'Winner'
epl_data['Winner']=np.where(epl_data['FTR'] == 'H',epl_data['HomeTeam'],epl_data['FTR'])
epl_data['Winner']=np.where(epl_data['FTR'] == 'A',epl_data['AwayTeam'],epl_data['Winner'])
epl_data['Winner']=np.where(epl_data['FTR'] == 'D',epl_data['AwayTeam']+' & '+epl_data['HomeTeam'],epl_data['Winner'])
#Finding the score of each winning match
epl_data['Score']=np.where(epl_data['FTHG'] > epl_data['FTAG'],epl_data['FTHG'],epl_data['FTAG'])
#Crosstabulation between the season and number of matches won by HomeTeam, AwayTeam and Season
df=pd.crosstab(index = epl_data["Season"],  columns=epl_data["FTR"],rownames=['Season']) 
df=pd.DataFrame(df);
#PLotting the above cross tabulation
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, ranges, LabelSet
from bokeh.plotting import figure
from bokeh.models import BoxSelectTool
from bokeh.io import push_notebook, output_notebook, curdoc
output_notebook()

source = ColumnDataSource(data=dict(df))


x=np.arange(25)\

p = figure(plot_width=1000, plot_height=600,title="Total number of winners in Away, Home and Draw Each Season",
          y_range=epl_data['Season'].unique())

p.hbar(y=np.arange(1,26)-0.7, height=0.2, left=0,
       right=df['A'], color="deepskyblue",legend="Away Team")
p.hbar(y=np.arange(1,26)-0.5, height=0.2, left=0,
       right=df['H'], color="darkcyan",legend="Home Team")
p.hbar(y=np.arange(1,26)-0.3, height=0.2, left=0,
       right=df['D'], color="blue",legend="Draw")
plot = figure(tools="pan,wheel_zoom,box_zoom,reset")
plot.add_tools(BoxSelectTool(dimensions="width"))

show(p)
#Cross tabulation between the Seasons and the Winners in each Season
df1=pd.crosstab(columns = epl_data["Winner"],  index=epl_data["Season"]) 
df1=pd.DataFrame(df1)
#Comparing the total number of goals scored by Arsenal and Chelsea in each Season
from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d
output_notebook()

# create a new plot with a range set with a tuple
p = figure(plot_width=600, plot_height=600, x_range=epl_data['Season'].unique()
           ,title="Arsenal and Chelsea Score over the years",)

# set a range using a Range1d
p.line(np.arange(1,26)-0.5, df1['Arsenal'], line_width=2,color="red")
p.circle(np.arange(1,26)-0.5, df1['Arsenal'], size=10,legend="Arsenal",color="red")

p.line(np.arange(1,26)-0.5, df1['Chelsea'], line_width=2,color="blue")
p.circle(np.arange(1,26)-0.5, df1['Chelsea'], size=10,legend="Chelsea",color="blue")

p.xaxis.major_label_orientation = "vertical"



show(p)
#Comparing the total number of goals scored by Man United and Chelsea in each Season

from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d

output_notebook()

# create a new plot with a range set with a tuple
p = figure(plot_width=600, plot_height=600, x_range=epl_data['Season'].unique(),title="Man Utd and Chelsea Score over the years",)

# set a range using a Range1d
p.line(np.arange(1,26)-0.5, df1['Man United'], line_width=2,color="red")
p.circle(np.arange(1,26)-0.5, df1['Man United'], size=10,legend="Man United",color="red")

p.line(np.arange(1,26)-0.5, df1['Chelsea'], line_width=2,color="blue")
p.circle(np.arange(1,26)-0.5, df1['Chelsea'], size=10,legend="Arsenal",color="blue")

p.xaxis.major_label_orientation = "vertical"

show(p)
#PLotting the total number of goals scored by Arsenal, Chelsea, Man United, Liverpool, Man City in each Season

from bokeh.layouts import gridplot
from bokeh.models import Range1d, HoverTool
from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d, HoverTool
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import ColumnDataSource 
output_notebook()

Man_Utd=df1.reset_index()['Man United'].tolist()
hover= HoverTool(tooltips=[("Total Goals scored","$y")])
# create a new plot
s1 = figure(x_range=epl_data['Season'].unique(),tools=[hover])
s1.circle(np.arange(1,26)-0.5, df1['Man United'].tolist(), size=5,legend="Man United",color="blue")
s1.line(np.arange(1,26)-0.5, df1['Man United'].tolist(), line_width=2,color="blue")

# create a new plot and share both ranges
s2 = figure(x_range=epl_data['Season'].unique(),tools=[hover])
s2.triangle(np.arange(1,26)-0.5, df1['Chelsea'].tolist(), size=5,legend="Chelsea",color="crimson")
s2.line(np.arange(1,26)-0.5, df1['Chelsea'].tolist(), line_width=2,color="crimson")

# create a new plot and share only one range\
s3 = figure(x_range=epl_data['Season'].unique(),tools=[hover])
s3.square(np.arange(1,26)-0.5, df1['Arsenal'].tolist(), size=5,legend="Arsenal",color="darkcyan")
s3.line(np.arange(1,26)-0.5, df1['Arsenal'].tolist(), line_width=2,color="darkcyan")

s4 = figure(x_range=epl_data['Season'].unique(),tools=[hover])
s4.square(np.arange(1,26)-0.5, df1['Liverpool'].tolist(), size=5,legend="Liverpool",color="red")
s4.line(np.arange(1,26)-0.5, df1['Liverpool'].tolist(), line_width=2,color="red")

s5 = figure(x_range=epl_data['Season'].unique(),tools=[hover])
s5.square(np.arange(1,26)-0.5, df1['Man City'].tolist(), size=5,legend="Man City",color="olivedrab")
s5.line(np.arange(1,26)-0.5, df1['Man City'].tolist(), line_width=2,color="olivedrab")


s1.xaxis.major_label_orientation = "vertical"
s2.xaxis.major_label_orientation = "vertical"
s3.xaxis.major_label_orientation = "vertical"
s4.xaxis.major_label_orientation = "vertical"
s5.xaxis.major_label_orientation = "vertical"

p = gridplot([[s1, s2, s3, s4, s5]], toolbar_location=None)


# show the results
show(p)
#Best Performance of different teams and the corresponding year
from matplotlib import pyplot
import matplotlib.pyplot as plt
result=epl_data['FTR'].tolist()
home_team=epl_data['HomeTeam'].tolist()
away_team=epl_data['AwayTeam'].tolist()
final_result=[]
for i in range(0,len(result)):
    if(result[i]=='A'):
        final_result.append(away_team[i])
    elif(result[i]=='H'):
        final_result.append(home_team[i])
    elif(result[i]=='D'):
        final_result.append('no result')


date=epl_data['Date'].tolist()
date_new=[]
for i in date:
    date_new.append(i.split('/')[-1])
date_new
final_date=[]
for i in date_new:
    if(len(i)==4):
        final_date.append(i[2:4])
    else:
        final_date.append(i)
len(final_result)
final_df=pd.DataFrame({'final_date':final_date,'final_result':final_result})
yearwise_win=final_df.groupby(['final_result','final_date']).size()
yearwise_win_df=yearwise_win.reset_index()
final_res_uniq=(list(set(final_result)))
y=yearwise_win_df.groupby(['final_result'])['final_date',0].max()
yearwise_win_final_df=y.reset_index()
yearwise_win_final_df
yearwise_win_df=yearwise_win_df.rename(columns = {0:'matches_won'})
max_count=yearwise_win_final_df[0].tolist()
max_count
yearwise_win_final_df.rename(columns = {0:'max_matches_won'})
c1=yearwise_win_df['final_result'].tolist()
c2=yearwise_win_final_df['final_result'].tolist()
d1=yearwise_win_df['matches_won'].tolist()
d2=max_count
e1=yearwise_win_df['final_date'].tolist()
d=[]
for i in range(0,51):
    for j in range(0,yearwise_win_df.shape[0]):
        if(c2[i]=='no result' and d2[i]==d1[j]):
            d.append(0)    
        elif(c2[i]==c1[j] and d2[i]==d1[j]):
            d.append(e1[j])
           
d_rem=[d[2],d[18],d[20],d[21],d[22],d[25]]

l=[2,18,20,21,22,25,30,45]
new_d= [i for j, i in enumerate(d) if j not in l]
needed_df=pd.DataFrame({'final_result':c2,'final_date':new_d,'max_matches_won':d2})
print(needed_df)
fig, ax = pyplot.subplots(figsize=(10,10))
plt.xticks(rotation=90)
sns.barplot(x='final_result',y='max_matches_won',data=needed_df,ax=ax)

#Following table represents the best performances of each team .i.e. 
#maximum number of matches they have won and season in which they have won
#Statistical test to check whether there is any association between half time result and full time result
# H0:The attributes half time results and full time results are independent
# H1:The attributes half time results and full time results are dependent

import scipy.stats
epl_data_half=epl_data.loc[np.arange(924,9664),:]
epl_data_half.tail()
a=epl_data_half['HTR']
b=epl_data_half['FTR']
obs=pd.crosstab(a, b, rownames=['half_time'], colnames=['full_time'])
print(obs)
g, p, dof, expctd = scipy.stats.chi2_contingency(obs, lambda_="log-likelihood")
g,p
 
#Since the p value is smaller than 0.05 so we reject our hypothesis at 5% level of significance
#and hence we can conclude that there is some dependence between the half time and full time results

