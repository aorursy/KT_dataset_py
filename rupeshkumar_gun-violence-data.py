import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib as mpl
%matplotlib inline
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
from plotly.offline import plot
print(plotly.__version__)
import calendar
plt.style.use("seaborn")
plt.style.use("seaborn")
import heapq, string, os, random
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import folium
from folium import plugins
from IPython.display import HTML, display
import collections 
from collections import Counter
GV=pd.read_csv("../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")
GV.info() # total usage

GV.memory_usage() # usage by column
GV.describe()
GV.head(5)
#Number of Rows
GV.shape[0]
#Number of Columns
GV.shape[1]
GV.index
GV.dtypes
#Column Names
GV.columns.values
GV.columns
GV['state'].value_counts()
#Arranging DateTime column into its component
GV['date']=pd.to_datetime(GV['date'])
GV.dtypes

GV['year'] = GV['date'].dt.year
GV['month'] = GV['date'].dt.month
GV['monthday'] = GV['date'].dt.day
GV['weekday'] = GV['date'].dt.weekday
GV.shape
GV['casualty']=GV['n_killed']+GV['n_injured']

#Segregating data Gender wise

GV["participant_gender"] = GV["participant_gender"].fillna("0::Unknown")
    
def gender(n) :                    
    gender_rows = []               
    gender_row = str(n).split("||")    
    for i in gender_row :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            gender_rows.append(g_row[1])    

    return gender_rows

gender_series = GV.participant_gender.apply(gender)
GV["total_participant"] = gender_series.apply(lambda x: len(x))
GV["male_participant"] = gender_series.apply(lambda i: i.count("Male"))
GV["female_participant"] = gender_series.apply(lambda i: i.count("Female"))
GV["unknown_participant"] = gender_series.apply(lambda i: i.count("Unknown"))
GV_null=GV.isnull().sum()
GV_dup=GV.duplicated().sum() # count of duplicates
GV_dup
GV_na=GV.isna().sum()
GV_nan=pd.concat([GV_null,GV_na],axis=1)
GV_nan

#Remove data not required 
GV.drop([
    "incident_url",
    "source_url",
    "incident_url_fields_missing",
    "sources"
], axis=1, inplace=True)

#year with maximum incidents recorded 
GV.year.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Gun Violence Incidents by year')
plt.ylabel('Number of incidents')
plt.xlabel('Year')
#Momth with  incidents recorded 
GV.month.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Gun Violence Incidents by year')
plt.ylabel('Number of incidents')
plt.xlabel('Month')
#weekday with incidents recorded 
GV.weekday.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Gun Violence Incidents by year')
plt.ylabel('Number of incidents')
plt.xlabel('Day')
#Total count Killed  Yearly due to Gun Violence

GV_yearly=GV.groupby(GV["year"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Killed, data=GV_yearly,label="yearly_vs_killed")
GV_yearly
#Total count Killed  Monthly due to Gun Violence

GV_yearly=GV.groupby(GV["month"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Killed, data=GV_yearly,label="yearly_vs_killed")
GV_yearly
#Total count Injured  Yearly due to Gun Violence

GV_yearly=GV.groupby(GV["year"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Injured, data=GV_yearly,label="yearly_vs_killed")
GV_yearly
#Total count Injured  Monthly due to Gun Violence

GV_yearly=GV.groupby(GV["month"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Injured, data=GV_yearly,label="yearly_vs_killed")
GV_yearly
import plotly
plotly.offline.init_notebook_mode() # run at the start of every ipython notebook
GV_cas = GV.reset_index().groupby(by=['state']).agg({'casualty':'sum', 'year':'count'}).rename(columns={'year':'count'})
GV_cas['state'] = GV_cas.index

trace1 = go.Bar(
    x=GV_cas['state'],
    y=GV_cas['count'],
    name='Number of Incidents',
)
trace2 = go.Bar(
    x=GV_cas['state'],
    y=GV_cas['casualty'],
    name='Total casualty',
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    margin=dict(b=150),
    legend=dict(dict(x=-.1, y=1.2)),
        )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#reference : https://github.com/amueller/word_cloud & https://github.com/amueller/word_cloud/blob/master/examples/masked.py

gun_mask = np.array(Image.open('../input/gungviol/gun_PNG1387.png'))
stopwords = set(STOPWORDS)
txt = " ".join(GV['gun_type'].dropna())
wc = WordCloud(mask=gun_mask, max_words=1200, stopwords=STOPWORDS, colormap='spring', background_color='Black').generate(txt)
plt.figure(figsize=(16,18))
plt.imshow(wc)
plt.axis('off')
plt.title('');

#reference : https://github.com/amueller/word_cloud & https://github.com/amueller/word_cloud/blob/master/examples/masked.py

gun_mask = np.array(Image.open("../input/usawcimg/USA-states (1).PNG"))
stopwords = set(STOPWORDS)
txt = " ".join(GV['location_description'].dropna())
wc = WordCloud(mask=gun_mask, max_words=1200, stopwords=STOPWORDS, colormap='spring', background_color='Black').generate(txt)
plt.figure(figsize=(16,18))
plt.imshow(wc)
plt.axis('off')
plt.title('');

GV['state'].value_counts().plot.pie(figsize=(20, 20), autopct='%.2f')
#Check for values to be displayed
plt.title("State wise pie diagram")
plt.ylabel('Number of State')
#  4. Statewise show dates with maximum incidents?- 

#Pie Chart
GV.state.value_counts().head().plot(kind = 'pie', figsize = (15,15))
plt.legend("state")
plt.title('Statewise distribution of incidents')
plt.xlabel('Number of incidents')
plt.ylabel('States')
#State with minimum incidents recorded 
GV.state.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Safest States in USA')
plt.ylabel('Number if incidents')
plt.xlabel('States')
state_s=pd.read_csv("../input/statesgv/states_GV.csv",index_col=0)

gun_killed = (GV[['state','n_killed']]
              .join(state_s, on='state')
              .groupby('Abbreviation')
              .sum()['n_killed']
             )
layout = dict(
        title = 'Safe State 2013-2018 ',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
data = [go.Choropleth(locationmode='USA-states',
             locations=gun_killed.index.values,
             text=gun_killed.index,
             z=gun_killed.values)]

fig = dict(data=data, layout=layout)

iplot(fig)
GV['guns'] = GV['n_guns_involved'].apply(lambda x : "5+" if x>=5 else str(x))

GV1 = GV['guns'].value_counts().reset_index()
GV1 = GV1[GV1['index'] != 'nan']
GV1 = GV1[GV1['index'] != '1.0']

labels = list(GV1['index'])
values = list(GV1['guns'])

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors = ['#blueviolet', '#magenta', '#96D38C', '#cyan', '#lime', '#orangered', '#k', '#b', '#aquamarine']))
layout = dict(height=600, title='Number of Guns Used', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)

GV_analysis = GV.sort_values(['casualty'], ascending=[False])
GV_analysis[['date', 'state', 'city_or_county', 'gun_type','n_killed', 'n_injured']].head(10)
Crimecount=GV['state'].value_counts().head(10)
Crimecount
plt.pie(Crimecount,labels=Crimecount.index,shadow=True)
plt.title("Top 10 High Crime Rate State")
plt.axis("equal")
state_D=pd.read_csv("../input/statessafe/states_D.csv",index_col=0)

gun_killed = (GV[['state','n_killed']]
              .join(state_D, on='state')
              .groupby('Abbreviation')
              .sum()['n_killed']
             )
layout = dict(
        title = 'To 10 Dangerous State 2013-2018 ',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
data = [go.Choropleth(locationmode='USA-states',
             locations=gun_killed.index.values,
             text=gun_killed.index,
             z=gun_killed.values)]

fig = dict(data=data, layout=layout)

iplot(fig)
#Ref : https://plot.ly/python/horizontal-bar-charts/

Types  = "||".join(GV['incident_characteristics'].dropna()).split("||")
incidents = Counter(Types).most_common(20)
inci1 = [x[0] for x in incidents]
inci2 = [x[1] for x in incidents]
trace1 = go.Scatter(
    x=inci2[::-2],
    y=inci1[::-2],
    name='Incident Report',
    marker=dict(color='rgba(50, 171, 96, 1.0)'),
    )
data = [trace1]
layout = go.Layout(
    barmode='overlay',
    margin=dict(l=350),
    width=900,
    height=600,
       title = 'Incident Report',
)

report = go.Figure(data=data, layout=layout)
iplot(report)

GV.state.value_counts().sort_index().plot(kind = 'barh', figsize = (20,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Statewise distribution of incidents')
plt.xlabel('Number of incidents')
plt.ylabel('States')
#State Vs No of People Killed
#no NAN in State and n_killed
sns.boxplot('state','n_killed',data=GV)
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=15)

#Violin plot analysis of number of killed and injured year wise
impact_numbers = GV[["n_killed","n_injured"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.violinplot(data=impact_numbers,split=True,inner="quartile")
## Box Plot for n_killed or n_injured for Chicago data.
sns.boxplot("n_killed", "n_injured", data= GV)
df1 = GV.sort_values(['casualty'], ascending=[False])
df1[['date', 'state', 'city_or_county', 'address', 'n_killed', 'n_injured']].head(10)
#REf : https://pythonhow.com/web-mapping-with-python-and-folium/


map_GV=GV[GV['n_killed'] >= 3][['latitude', 'longitude', 'casualty', 'n_killed']].dropna()
m1 = folium.Map([39.50, -98.35], tiles='CartoDB dark_matter', zoom_start=3.5)
#m2 = folium.Map([39.50, -98.35], zoom_start=3.5, tiles='cartodbdark_matter')
markers=[]
for i, row in map_GV.iterrows():
    casualty = row['casualty']
    if row['casualty'] > 100:
        casualty = row['casualty']*0.1 
    folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=float(casualty), color='#blue', fill=True).add_to(m1)
m1
#Number of person killed vs incident
sns.jointplot("incident_id",
             "n_killed",
             GV,
             kind="scatter",
             s=100, color="m",edgecolor="blue",linewidth=2)
#Swarm plot analysis of number of killed and number of guns involved
impact_numbers = GV[["n_killed","n_guns_involved"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.swarmplot(x="n_killed",y="n_guns_involved",data=impact_numbers)
#Factor plot analysis of number of killed and injured year wise
impact_numbers = GV[["n_injured","n_guns_involved"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.factorplot(data=impact_numbers,split=True,inner="quartile")
#Violin plot analysis of number of killed and injured year wise
impact_numbers = GV[["total_participant"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.violinplot(data=impact_numbers,split=True)
# Plot using Seaborn
sns.lmplot(x='n_killed', y='n_injured', data=GV,
           fit_reg=False, 
           hue='state')
 
# Tweak using Matplotlib
plt.ylim(0, None)
plt.xlim(0, None)
#Number of person injured vs incident
sns.jointplot("incident_id",
             "n_injured",
             GV,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)
#Number of Male vs killed

sns.jointplot("male_participant",
             "n_killed",
             GV,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)
#Density plot for yearly incident 
yearly_casulaty = GV[["n_killed", "n_injured"]].groupby(GV["year"]).sum()
d_plot=sns.kdeplot(yearly_casulaty['n_killed'],shade=True,color="b")
d_plot=sns.kdeplot(yearly_casulaty['n_injured'],shade=True,color="g")
del(yearly_casulaty)
yearly_actor = GV[["total_participant","male_participant", "female_participant"]].groupby(GV["year"]).sum()
density_plot=sns.kdeplot(yearly_actor['total_participant'],shade=True,color="r")
density_plot=sns.kdeplot(yearly_actor['male_participant'],shade=True,color="g")
density_plot=sns.kdeplot(yearly_actor['female_participant'],shade=True,color="k")
del(yearly_actor)
sns.distplot(GV.year)
g = sns.FacetGrid(GV, col="year", col_wrap=4, ylim=(0, 10))
g.map(sns.pointplot, "male_participant", "n_killed", color=".3", ci=None);
g = sns.FacetGrid(GV, col="year", col_wrap=4, ylim=(0, 10))
g.map(sns.pointplot, "female_participant", "n_killed", color=".3", ci=None);
g = sns.FacetGrid(GV, col="year", col_wrap=4, ylim=(0, 10))
g.map(sns.pointplot, "unknown_participant", "n_killed", color=".3", ci=None);
sns.boxplot([GV.month, GV.n_injured])
sns.boxplot([GV.month, GV.n_killed])
g = sns.FacetGrid(GV, col="year", aspect=.5)
g.map(sns.barplot, "n_injured", "weekday")
g = sns.FacetGrid(GV, col="year", aspect=.5)
g.map(sns.barplot, "n_killed", "weekday")
g = sns.FacetGrid(GV, col="year")
g.map(plt.hist, "male_participant");
sns.set(style="ticks")
g = sns.FacetGrid(GV, row="n_killed", col="year", margin_titles=True)
g.map(sns.regplot, "year", "n_guns_involved", color=".3", fit_reg=False, x_jitter=.1);
g = sns.FacetGrid(GV, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","+"]})
g.map(plt.scatter, "male_participant", "female_participant", s=100, linewidth=.10, edgecolor="black")
g.add_legend();
