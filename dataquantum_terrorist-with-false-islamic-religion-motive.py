from IPython.display import Image 
from IPython.core.display import HTML 
Image(url= "https://i.guim.co.uk/img/media/f0ae22be2f514dc6613290ba528491c2d4862d99/0_0_1763_2351/master/1763.png?width=700&quality=85&auto=format&fit=max&s=12c1972af47cfccc7be40228ca104f76")
import pandas as pd
import codecs
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set()
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import squarify
from plotly.offline import iplot, init_notebook_mode
py.init_notebook_mode(connected=True)
import squarify

terrorist=pd.DataFrame(pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1'))
terrorist.shape
terrorist.info()
terrorist.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terrorist.head()
islam = terrorist['Motive'].str.contains('Islam', na=False).astype(int)
islam_df = terrorist[terrorist['Motive'].str.contains('Islam', regex=False, case=False, na=False)]
islam_df.head()
islam_df.info()
islam_df.shape
muslim = terrorist['Motive'].str.contains('Muslim', na=False).astype(int)
muslim_df = terrorist[terrorist['Motive'].str.contains('Muslim', regex=False, case=False, na=False)]
muslim_df.head()
muslim_df.info()
muslim_df.shape
extremist_islam = islam.sum() + muslim.sum()
extremist_islam
percentage_islam = round((extremist_islam)/terrorist['eventid'].count()*100, 2) 
print("Islam/Muslim Terrorism : " + str(percentage_islam) + "%")
labels = ['Islam/Muslim Terrorism','Others Group of Terrorism']
values = [percentage_islam, 100-percentage_islam]
trace = go.Pie(labels=labels, values=values)
title ='Muslim Terrorism Percentage'
py.iplot([trace], title, filename='basic_pie_chart')

total_muslim = 1800000000
percentage_extremist = round(extremist_islam/total_muslim*100, 10)
print("Percentage terrorism from Muslim Population : " + str(percentage_extremist) + " %")
motive = terrorist['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
wordcloud = WordCloud().generate(motive)
plt.subplots(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.subplots(figsize=(20,10))
sns.countplot('Year',data=terrorist,palette="husl",edgecolor=sns.color_palette('dark',8))
plt.xticks(rotation=90)
plt.suptitle('All Terrorist Cases (1970 - 2017)')
plt.show()
t = "Out of top 15 countries most affected by terrorism, Six of them is Muslim Country"
plt.subplots(figsize=(18,10))
sns.barplot(terrorist['Country'].value_counts()[:15].index,terrorist['Country'].value_counts()[:15].values,palette="husl")
plt.suptitle('Top Affected Countries')
plt.xticks (rotation = 45)
plt.text(7, 20000, t, ha='left', va='top', style='italic', rotation=0, wrap=True)
plt.show()
muslim_terror = pd.concat([islam_df, muslim_df], ignore_index=True)
muslim_terror.shape
muslim_terror.head()
muslim_terror['Motive'].head(10)
motive = muslim_terror['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
wordcloud = WordCloud().generate(motive)
plt.subplots(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
t1 = "There is significant of increase of Muslim Terrorism in 2012. But why? Arab Spring ? "
plt.subplots(figsize=(15,10))
sns.countplot('Year',data=muslim_terror,palette="husl",edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.text(17, 300, t1, ha='right', va='bottom', style='italic', rotation=0, wrap=True)
plt.suptitle('Number Of Muslim Terrorism Activities Each Year')
plt.show()
t2 = "Out of top 15 countries most affected by Muslim terrorism, Ten of them is Muslim Country"
plt.subplots(figsize=(18,10))
sns.barplot(muslim_terror['Country'].value_counts()[:15].index,muslim_terror['Country'].value_counts()[:15].values,palette="husl")
plt.suptitle('Top Affected Countries of Muslim Terror')
plt.text(5, 450, t2, ha='left', va='top', style='italic', rotation=0, wrap=True)
plt.xticks(rotation =45)
plt.show()
sns.barplot(terrorist['Group'].value_counts()[1:15].values,terrorist['Group'].value_counts()[1:15].index,palette=('husl'))
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.suptitle('All Terrorist Groups with Highest Terror Attacks')
plt.show()
sns.barplot(muslim_terror['Group'].value_counts()[1:15].values,muslim_terror['Group'].value_counts()[1:15].index,palette=('husl'))
plt.xticks(rotation=90)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.suptitle('Muslim Terrorist Groups with Highest Terror Attacks')
plt.show()
region = muslim_terror.groupby('Region', as_index=False)['Year'].sum()
plt.subplots(figsize=(10,10))
squarify.plot(sizes=region.iloc[:,1], label=region.iloc[:,0], alpha=.75 )
plt.axis('on')
plt.show()
plt.subplots(figsize=(15,10))
sns.countplot('Region',data=muslim_terror,palette='husl',edgecolor=sns.color_palette('dark',7),order=muslim_terror['Region'].value_counts().index)
plt.xticks(rotation=90)
plt.suptitle('Number Of Terrorist with Islamic Religion Motive By Region')
plt.show()
muslim_terror = muslim_terror.fillna(0)
muslim_terror['Day'][muslim_terror.Day == 0] = 1
muslim_terror['date'] = pd.to_datetime(muslim_terror[['Day', 'Month', 'Year']])
muslim_terror['text'] = ' Date : ' + muslim_terror['date'].dt.strftime('%B %-d, %Y') + '<br>' +\
                        ' Country : ' + muslim_terror['Country'].astype(str)  + '<br>' +\
                        ' Group : ' + muslim_terror['Group'].astype(str)  + '<br>' +\
                        ' Killed : ' + muslim_terror['Killed'].astype(str)  + '<br>' +\
                        ' Injured : ' + muslim_terror['Wounded'].astype(str)

kill =  dict(
        type = 'scattergeo',
        text = muslim_terror[muslim_terror.Killed > 0]['text'],
        autocolorscale = True,
        reversescale = True,
        name = 'Killed',
        locationmode = muslim_terror['Country'],
           lon = muslim_terror[muslim_terror.Killed > 0]['longitude'],
           lat = muslim_terror[muslim_terror.Killed > 0]['latitude'],
           mode = 'markers',
        marker = dict(
               size = muslim_terror[muslim_terror.Killed > 0]['Killed'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'red')
       
      ) 

wound =  dict(
        type = 'scattergeo',
        text =  muslim_terror[muslim_terror.Wounded > 0]['text'],
        autocolorscale = True,
        reversescale = True,
        name = 'Wounded',
        locationmode = muslim_terror['Country'],
           lon = muslim_terror[muslim_terror.Wounded > 0]['longitude'],
           lat = muslim_terror[muslim_terror.Wounded > 0]['latitude'],
           mode = 'markers',
        marker = dict(
               size = muslim_terror[muslim_terror.Wounded > 0]['Wounded'] ** 0.255 * 8,
               opacity = 0.1,
               color = 'yellow')
       
      ) 

layout = dict(
         title = 'Islamic Terrorist Attacks by Latitude/Longitude (1970-2017)',
         showlegend = True,
         legend = dict(
             x = 0.1, y = 0.1
         ),
         geo = dict(
             showland = True,
             landcolor = 'rgb(217, 217, 217',
             subunitwidth = 1,
             subunitcolor="rgb(255, 255, 255)",
             countrywidth = 1,
             countrycolor="rgb(255, 255, 255)",
             showlakes = True,
             lakecolor = 'bluesky')
         )
data = [kill, wound]
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-populations' )
othersterror=pd.DataFrame(terrorist.groupby('Year',as_index=False)['Killed'].sum())
muslimterror=pd.DataFrame(muslim_terror.groupby('Year', as_index=False)['Killed'].sum())
kill_cases = pd.merge(othersterror, muslimterror, on = 'Year',how='outer', suffixes=(' by All terrorist', ' by terrorist with Islamic religion motive'))
kill_cases.fillna(0)
kill_cases = pd.DataFrame(kill_cases)
kill_cases.head()
count_kill=muslim_terror.groupby('Year')['Killed'].sum().to_frame()
count_kill.plot.bar(width=0.9, color='green')
fig=plt.gcf()
fig.set_size_inches(18,10)
plt.suptitle('Killed cases by terrorist with Islamic Religion motive')
plt.show()
trace1 = go.Bar(
    x=kill_cases['Year'],
    y=kill_cases.iloc[:,2],
    name='Muslim Terrorism',

)
trace2 = go.Bar(
    x=kill_cases['Year'],
    y=kill_cases.iloc[:,1],
    name='All Terrorism'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
attack = muslim_terror.groupby('AttackType', as_index=False)['Year'].sum()
plt.subplots(figsize=(10,10))
squarify.plot(sizes=attack.iloc[:,1], label=attack.iloc[:,0], alpha=.75 )
plt.axis('on')
plt.show()
plt.subplots(figsize=(15,10))
sns.countplot('AttackType',data=muslim_terror,palette='husl',order=muslim_terror['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.suptitle('Attacking Methods by Muslim Terrorists')
plt.show()
target = muslim_terror.groupby('Target_type', as_index=False)['Year'].sum()
plt.subplots(figsize=(10,10))
squarify.plot(sizes=target.iloc[:,1], label=target.iloc[:,0], alpha=.75 )
plt.axis('on')
plt.show()
plt.subplots(figsize=(15,10))
sns.countplot(muslim_terror['Target_type'],palette='husl',order=muslim_terror['Target_type'].value_counts().index)
plt.xticks(rotation=90)
plt.suptitle('Favorite Targets of Terrorist with Islamic Religion Motive')
plt.show()
terror_region=pd.crosstab(muslim_terror.Year,muslim_terror.Region)
terror_region.plot(color=sns.color_palette('Set1',12))
fig=plt.gcf()
fig.set_size_inches(18,10)
plt.suptitle('Trend of Terrorist Activities with Islamic Religion Motive')
plt.show()
top_groups10=muslim_terror[muslim_terror['Group'].isin(muslim_terror['Group'].value_counts()[1:11].index)]
pd.crosstab(top_groups10.Year,top_groups10.Group).plot(color=sns.color_palette('Paired',10))
fig=plt.gcf()
fig.set_size_inches(18,10)
plt.show()

