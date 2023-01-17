# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import pycountry
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.info()
df.isnull().sum()
df['Year']=df['Year'].fillna(method='ffill')
df.drop(columns='Rank',inplace=True)
df['Year']=df['Year'].astype(int)
alltimesales =round(df.groupby(["Genre"]).mean(),2)
alltimesales.reset_index(inplace=True)
fig = px.bar(alltimesales.sort_values('Global_Sales'), 
             x='Global_Sales', y='Genre', title='Average Global Sales(million dollars) across years for different Game Genres'
             , text='Global_Sales', orientation='h', 
             width=1000, height=500, range_x = [0,1],color_discrete_sequence=px.colors.qualitative.D3)
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.show()
platform=pd.DataFrame(df['Platform'].value_counts())
platform.reset_index(inplace=True)
platform.rename(columns={'index':'Platform','Platform':'Number of Games'},inplace=True)
fig = px.bar(platform.sort_values('Number of Games'), 
             x='Number of Games', y='Platform', title='Number of games across different platforms', 
             text='Number of Games', orientation='h', 
             width=1000, height=800, range_x = [0,2200],color_discrete_sequence=px.colors.qualitative.D3,template='presentation')
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.show()
platformsalesacrossyears= df[["Year","Platform",'Global_Sales']]
platformsalesacrossyears['total_sales'] =platformsalesacrossyears.groupby(['Platform','Year'])['Global_Sales'].transform('sum')
platformsalesacrossyears
yearlyreleases=pd.DataFrame(df['Year'].value_counts())
yearlyreleases.reset_index(inplace=True)
yearlyreleases.rename(columns={'index':'Year','Year':'Number of Releases'},inplace=True)
fig = px.bar(yearlyreleases.sort_values('Year',ascending=True), 
             x='Number of Releases', y='Year', title='Number of games released in each year', 
             text='Number of Releases', orientation='h', 
             width=1000, height=1000, range_x = [0,1500],color_discrete_sequence=px.colors.qualitative.D3,template='presentation')
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.show()
genre = df.loc[:,['Genre','Global_Sales']]
genre['total_sales'] = genre.groupby('Genre')['Global_Sales'].transform('sum')
genre
platformsales=df[['Platform','Year','Global_Sales']]
platformsales['Netplatformsales']=platformsales.groupby(['Platform','Year'])['Global_Sales'].transform('sum')
desiredplatforms=['DS','PS2','PS3','Wii','X360','PSP','PS','PC','XB','GBA']

platformsales=platformsales[(platformsales['Year']>=1998) & (platformsales['Year']<=2015) & (platformsales['Platform'].isin(desiredplatforms))]
platformsales=platformsales.sort_values('Year',ascending=True)
platformsales.drop_duplicates()
platformsales.drop('Global_Sales', axis=1, inplace=True)
fig=px.bar(platformsales,x='Platform', y='Netplatformsales', animation_frame="Year",range_y=[0,220], 
           animation_group='Netplatformsales', hover_name='Platform',color_discrete_sequence=px.colors.qualitative.D3,
          title='Change in Net Sales(million $) across different platforms from 1998 to 2015')
fig.show()
salespergenre=df.groupby('Genre').sum()
salespergenre['percentagesales']=round(salespergenre['Global_Sales']/(salespergenre['Global_Sales'].sum()),4)*100
salespergenre.reset_index(inplace=True)
fig = px.pie(salespergenre, names='Genre', values='percentagesales',
            color_discrete_sequence=px.colors.qualitative.D3,
            template='presentation')
fig.update_traces(rotation=90,pull=[0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06],textfont_size=20,
                 marker=dict(line=dict(color='#000000', width=1)),textinfo="percent+label")
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.show()
labels = list(salespergenre['Genre'])
values = list(salespergenre['percentagesales'])
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_traces(rotation=160,pull=[0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06],textfont_size=20,
                 marker=dict(line=dict(color='#000000', width=1)),textinfo='percent+label')
fig.show()
toppublishers=pd.DataFrame(df['Publisher'].value_counts())
toppublishers.reset_index(inplace=True)
toppublishers.rename(columns={'index':'Publisher','Publisher':'Number of Releases'},inplace=True)
toppublishers.head(10)
publishersales=df[['Publisher','Year','Global_Sales']]
publishersales['Netpublishersales']=publishersales.groupby(['Publisher','Year'])['Global_Sales'].transform('sum')
desiredpublisher=['Electronic Arts','Activison','Namco Bandal Games','Ubisoft','Konami Digital Entertainment',
                  'THQ','Nintendo','Sega','Sony Computer Entertainment','Take-Two Interactive']

publishersales=publishersales[(platformsales['Year']>=1998) & (platformsales['Year']<=2015) & (publishersales['Publisher'].isin(desiredpublisher))]
publishersales=publishersales.sort_values('Year',ascending=True)
publishersales.drop_duplicates()
publishersales.drop('Global_Sales', axis=1, inplace=True)
fig=px.bar(publishersales,x='Publisher', y='Netpublishersales', animation_frame="Year",range_y=[0,210], 
           animation_group="Netpublishersales", hover_name="Publisher",color_discrete_sequence=px.colors.qualitative.D3,width=1000, height=500
          ,title='Change in Net Sales(million $) of different Publishers from 1998 to 2015')
fig.show()
areasales=df
areasales[['North America Sales','Europe Sales','Japan Sales','Sales in Other Regions']]=areasales.groupby('Year')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].transform('sum')
areasales
areasales.drop(['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Genre','Platform','Publisher','Global_Sales','Name'], axis=1, inplace=True)
areasales
areasales=areasales[(areasales['Year']>=1985) & (areasales['Year']<=2015)]
areasales=areasales.sort_values('Year',ascending=True)
areasales.set_index('Year',inplace=True)
areasales.drop_duplicates(inplace=True)
areasales.style.background_gradient(cmap='Blues',axis=1)
df2=pd.read_csv('../input/videogamesales/vgsales.csv')
df2['Year']=df['Year'].fillna(method='ffill')
df2.drop(columns='Rank',inplace=True)
df2['Year']=df['Year'].astype(int)
action=df2[df2['Genre']=='Action']
sports=df2[df2['Genre']=='Sports']
misc=df2[df2['Genre']=='Misc']
roleplaying=df2[df2['Genre']=='Role-Playing']
shooter=df2[df2['Genre']=='Shooter']
racing=df2[df2['Genre']=='Racing']
platform=df2[df2['Genre']=='Platform']
simulation=df2[df2['Genre']=='Simulation']
fighting=df2[df2['Genre']=='Fighting']
strategy=df2[df2['Genre']=='Strategy']
puzzle=df2[df2['Genre']=='Puzzle']
action['Net_sales']=action.groupby(['Name'])['Global_Sales'].transform('sum')
action.sort_values('Net_sales',ascending=False,inplace=True)
action5=action[['Name','Net_sales']]
action5=action5.drop_duplicates().head(5)
action5
sports['Net_sales']=sports.groupby(['Name'])['Global_Sales'].transform('sum')
sports.sort_values('Net_sales',ascending=False,inplace=True)
sports5=sports[['Name','Net_sales']]
sports5=sports5.drop_duplicates().head(5)
sports5
misc['Net_sales']=misc.groupby(['Name'])['Global_Sales'].transform('sum')
misc.sort_values('Net_sales',ascending=False,inplace=True)
misc5=misc[['Name','Net_sales']]
misc5=misc5.drop_duplicates().head(5)
misc5
roleplaying['Net_sales']=roleplaying.groupby(['Name'])['Global_Sales'].transform('sum')
roleplaying.sort_values('Net_sales',ascending=False,inplace=True)
roleplaying5=roleplaying[['Name','Net_sales']]
roleplaying5=roleplaying5.drop_duplicates().head(5)
roleplaying5
shooter['Net_sales']=shooter.groupby(['Name'])['Global_Sales'].transform('sum')
shooter.sort_values('Net_sales',ascending=False,inplace=True)
shooter5=shooter[['Name','Net_sales']]
shooter5=shooter5.drop_duplicates().head(5)
shooter5
racing['Net_sales']=racing.groupby(['Name'])['Global_Sales'].transform('sum')
racing.sort_values('Net_sales',ascending=False,inplace=True)
racing5=racing[['Name','Net_sales']]
racing5=racing5.drop_duplicates().head(5)
racing5
platform['Net_sales']=platform.groupby(['Name'])['Global_Sales'].transform('sum')
platform.sort_values('Net_sales',ascending=False,inplace=True)
platform5=platform[['Name','Net_sales']]
platform5=platform5.drop_duplicates().head(5)
platform5
simulation['Net_sales']=simulation.groupby(['Name'])['Global_Sales'].transform('sum')
simulation.sort_values('Net_sales',ascending=False,inplace=True)
simulation5=simulation[['Name','Net_sales']]
simulation5=simulation5.drop_duplicates().head(5)
simulation5
fighting['Net_sales']=fighting.groupby(['Name'])['Global_Sales'].transform('sum')
fighting.sort_values('Net_sales',ascending=False,inplace=True)
fighting5=fighting[['Name','Net_sales']]
fighting5=fighting5.drop_duplicates().head(5)
fighting5
strategy['Net_sales']=strategy.groupby(['Name'])['Global_Sales'].transform('sum')
strategy.sort_values('Net_sales',ascending=False,inplace=True)
strategy5=strategy[['Name','Net_sales']]
strategy5=strategy5.drop_duplicates().head(5)
strategy5
puzzle['Net_sales']=puzzle.groupby(['Name'])['Global_Sales'].transform('sum')
puzzle.sort_values('Net_sales',ascending=False,inplace=True)
puzzle5=puzzle[['Name','Net_sales']]
puzzle5=puzzle5.drop_duplicates().head(5)
puzzle5
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 5 Games in Sports Genre','Top 5 Games in Sports Genre'))
fig.add_trace(go.Bar( y=action5['Net_sales'], x=action5['Name'],  
                     marker=dict(color=action5['Net_sales'], coloraxis="coloraxis")),
              1, 1)
                     
fig.add_trace(go.Bar( y=sports5['Net_sales'], x=sports5['Name'],  
                     marker=dict(color=sports5['Net_sales'], coloraxis="coloraxis")),
              1, 2)                     
                    
fig.update_layout(coloraxis=dict(colorscale='blues'), showlegend=False,plot_bgcolor='rgb(250, 242, 242)')
fig.show()
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 5 Games in Misc Genre','Top 5 Games in Role-Playing Genre',))



fig.add_trace(go.Bar( y=misc5['Net_sales'], x=misc5['Name'],  
                     marker=dict(color=misc5['Net_sales'], coloraxis="coloraxis")),
              1, 1)
                     
fig.add_trace(go.Bar( y=roleplaying5['Net_sales'], x=roleplaying5['Name'],  
                     marker=dict(color=roleplaying5['Net_sales'], coloraxis="coloraxis")),
              1, 2)  




fig.update_layout(coloraxis=dict(colorscale='blues'), showlegend=False,plot_bgcolor='rgb(250, 242, 242)')
fig.show()
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 5 Games in Shooter Genre','Top 5 Games in Racing Genre',))

fig.add_trace(go.Bar( y=shooter5['Net_sales'], x=shooter5['Name'],  
                     marker=dict(color=shooter5['Net_sales'], coloraxis="coloraxis")),
              1, 1)
                     
fig.add_trace(go.Bar( y=racing5['Net_sales'], x=racing5['Name'],  
                     marker=dict(color=racing5['Net_sales'], coloraxis="coloraxis")),
              1, 2)  

fig.update_layout(coloraxis=dict(colorscale='blues'), showlegend=False,plot_bgcolor='rgb(250, 242, 242)')
fig.show()
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 5 Games in Platform Genre','Top 5 Games in Simulation Genre',))

fig.add_trace(go.Bar( y=platform5['Net_sales'], x=platform5['Name'],  
                     marker=dict(color=shooter5['Net_sales'], coloraxis="coloraxis")),
              1, 1)
                     
fig.add_trace(go.Bar( y=simulation5['Net_sales'], x=simulation5['Name'],  
                     marker=dict(color=racing5['Net_sales'], coloraxis="coloraxis")),
              1, 2)  

fig.update_layout(coloraxis=dict(colorscale='blues'), showlegend=False,plot_bgcolor='rgb(250, 242, 242)')
fig.show()
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 5 Games in Fighter Genre','Top 5 Games in Strategy Genre',))

fig.add_trace(go.Bar( y=fighting5['Net_sales'], x=fighting5['Name'],  
                     marker=dict(color=shooter5['Net_sales'], coloraxis="coloraxis")),
              1, 1)
                     
fig.add_trace(go.Bar( y=strategy5['Net_sales'], x=strategy5['Name'],  
                     marker=dict(color=racing5['Net_sales'], coloraxis="coloraxis")),
              1, 2)  

fig.update_layout(coloraxis=dict(colorscale='blues'), showlegend=False,plot_bgcolor='rgb(250, 242, 242)')
fig.show()

fig=px.bar(puzzle5, y=puzzle5['Net_sales'], x=puzzle5['Name'],  
            color_discrete_sequence=px.colors.qualitative.D3,width=1000, height=500
          ,title='Top 5 Games in Puzzle Genre')
fig.show()
df2['nasales']=df2.groupby('Year')['NA_Sales'].transform(sum)
nasales=df2[['Year','nasales']]
nasales.insert(2,'Country',value=['North America']*len(nasales))

df2['jpsales']=df2.groupby('Year')['JP_Sales'].transform(sum)
jpsales=df2[['Year','jpsales']]
jpsales.insert(2,'Country',value=['Japan']*len(jpsales))

df2['eusales']=df2.groupby('Year')['EU_Sales'].transform(sum)
eusales=df2[['Year','eusales']]
eusales.insert(2,'Country',value=['Europe']*len(eusales))

df2['otsales']=df2.groupby('Year')['Other_Sales'].transform(sum)
othersales=df2[['Year','otsales']]
othersales.insert(2,'Country',value=['Other Countreis']*len(othersales))
tfdata = pd.concat([nasales,jpsales,eusales,othersales], axis=0)
tfdata.fillna(value=0,inplace=True)
tfdata['Net Sales']=tfdata['eusales']+tfdata['jpsales']+tfdata['nasales']+tfdata['otsales']
tfdata
fig=px.bar(tfdata.sort_values('Year',ascending=True),x='Country', y='Net Sales', animation_frame="Year",range_y=[0,400], 
           animation_group='Net Sales',hover_name='Country',color_discrete_sequence=px.colors.qualitative.D3,width=1000, height=500
          ,title='Net sales in different countries from 1980 to 2016')
fig.show()
