# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.templates.default="plotly_dark"
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import warnings
warnings.filterwarnings('ignore')
testinglab=pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv',encoding='utf-8')
testinglab.head()
India_coor=pd.read_csv('/kaggle/input/indian-coordinates/Indian Coordinates.csv')
India_coor.head()
testinglab.type.unique()
testinglab['type'].count()
testinglab.info()
sns.countplot(x=testinglab['type'],data=testinglab)
dbd_India=pd.read_excel('/kaggle/input/per-day-cases/per_day_cases.xlsx',sheet_name='India')
dbd_India.head()
dbd_Italy=pd.read_excel('/kaggle/input/per-day-cases/per_day_cases.xlsx',sheet_name='Italy')
dbd_Italy.head()
dbd_Korea=pd.read_excel('/kaggle/input/per-day-cases/per_day_cases.xlsx',sheet_name='Korea')

dbd_Korea.head()
India_coor.rename(columns={
    'Name of State / UT':'state',
    'Latitude':'lat',
    'Longitude':'long'
},inplace=True)
testinglab_full=pd.merge(testinglab,India_coor,on='state')
testinglab_full
train=pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
train.head()
train.info()
x=train.groupby('AgeGroup')['TotalCases'].sum().sort_values(ascending=False).to_frame()
x.style.background_gradient(cmap='Blues')
fig=px.bar(train[['AgeGroup','TotalCases']].sort_values('TotalCases',ascending=False),y='TotalCases',x='AgeGroup',color='TotalCases',log_y=False,template='ggplot2',title='AgeGroup vs total cases')
fig.show()
import seaborn as sns
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="AgeGroup", y="Percentage", data=train,
             color="blue")
plt.plot(train.AgeGroup,train.Percentage,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in train',fontsize=70)
plt.show()

plt.figure(figsize=(100,30))
plt.bar(train.AgeGroup, train.Percentage,label="Train")
plt.xlabel('AgeGroup')
plt.ylabel("Percentage")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in train',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()
plt.figure(figsize=(15,15))
plt.title('AgeGroup details by percentage',fontsize=20)
plt.pie(train['TotalCases'],autopct='%1.1f%%')
plt.legend(train['TotalCases'],loc='best')
plt.show()
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="AgeGroup", y="TotalCases", data=train,
             color="blue")
plt.plot(train.AgeGroup,train.TotalCases,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in train',fontsize=70)
plt.show()

plt.figure(figsize=(100,30))
plt.bar(train.AgeGroup, train.TotalCases,label="Train")
plt.xlabel('AgeGroup')
plt.ylabel("TotalCases")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in train',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()
data=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',parse_dates=['Date'])
data.head()
data.tail(10)
data.info()
print(f"Earliest Entry:{data['Date'].min()}")
print(f"Last Entry:{data['Date'].max()}")
print(f"Total Days: {data['Date'].max()-data['Date'].min()}")
data.rename(columns={'Date':'date',
                    'Sno':'sno',
                     'State/UnionTerritory':'state',
                     'ConfirmedIndianNational':'CIN',
                     'ConfirmedForeignNational':'CFN',
                     'Cured':'cured',
                     'Deaths':'deaths'
                    },inplace=True)
data.head()  
data.drop(columns=['sno'],axis=1,inplace=True)

data['Cases']=data['cured']+data['deaths']
data['Active Cases']=data['Confirmed']-data['Cases']


data.head(20)

data.tail(20)
data['Confirmed'].sum()
data['deaths'].sum()
data['cured'].sum()
x=data.groupby('state')['Active Cases'].sum().sort_values(ascending=False).to_frame()
x.style.background_gradient(cmap='Blues')
x=data.groupby('state')['deaths'].sum().sort_values(ascending=False).to_frame()
x.style.background_gradient(cmap='Blues')
fig=px.bar(data.sort_values('deaths',ascending=False).sort_values('deaths',ascending=True),x="deaths",y="state",title='Total Deaths Cases',text='deaths',orientation='h',width=1000,height=700,range_x=[0,max(data['deaths'])])
fig.update_traces(marker_color='#55ff45',opacity=0.8,textposition='inside')
fig.update_layout(plot_bgcolor='rgb(245,242,242)')
fig.show()
fig=px.bar(data.sort_values('Active Cases',ascending=False).sort_values('Active Cases',ascending=True),x="deaths",y="state",title='Total Active Cases',text='deaths',orientation='h',width=1000,height=700,range_x=[0,max(data['Active Cases'])])
fig.update_traces(marker_color='#55ff45',opacity=0.8,textposition='inside')
fig.update_layout(plot_bgcolor='rgb(245,242,242)')
fig.show()
fig=px.bar(data.sort_values('cured',ascending=False).sort_values('cured',ascending=True),x="cured",y="state",title='Total cured Cases',text='cured',orientation='h',width=1000,height=700,range_x=[0,max(data['deaths'])])
fig.update_traces(marker_color='#55ff45',opacity=0.8,textposition='inside')
fig.update_layout(plot_bgcolor='rgb(245,242,242)')
fig.show()
#latest
latest=data[data['date']==max(data['date'])].reset_index()
Kerala_latest=latest[latest['state']=='Kerala']
row_latest=latest[latest['state']!='Kerala']
#latest condensed
latest_grouped=latest.groupby('state')['cured','deaths','Confirmed','Active Cases'].sum().reset_index()
Kerala_latest_grouped=Kerala_latest.groupby('state')['cured','deaths','Confirmed','Active Cases'].sum().reset_index()
row_latest_grouped=row_latest.groupby('state')['cured','deaths','Confirmed','Active Cases'].sum().reset_index()
#latest Complete Data
temp=data.groupby(['state'])['cured','deaths','Confirmed','Active Cases'].max()
temp=data.groupby('date')['cured','deaths','Confirmed','Active Cases'].sum().reset_index()
temp=temp[temp['date']==max(temp['date'])].reset_index(drop=True)
temp.style.background_gradient(cmap='Pastel1')


temp=data.groupby(['state'])['cured','deaths','Active Cases'].max()
temp=data.groupby('date')['cured','deaths','Active Cases'].sum().reset_index()
temp=temp[temp['date']==min(temp['date'])].reset_index(drop=True)
temp.style.background_gradient(cmap='Pastel1')
temp_1=latest_grouped.sort_values(by='deaths',ascending=False)
temp_1=temp_1.reset_index(drop=True)
temp_1.style.background_gradient(cmap='Blues')
India_coor.head()
data_full=pd.merge(data,India_coor,on='state')
data_full
temp=data_full.groupby('date')['cured','deaths','Active Cases'].sum().reset_index()
temp=temp.melt(id_vars='date',value_vars=['cured','deaths','Active Cases'],var_name='Case',value_name='Count')

temp.head()
fig=px.area(temp,x="date",y="Count",color="Case",title="Cases Over Time",color_discrete_sequence=['#ffeebb',"#2367ff","#556677"])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
import folium
data_full=pd.merge(India_coor,data,on='state')
map=folium.Map(locations=[20,80],zoom_start=3.5,tiles='Stamen Toner')
for lat,lon,value,name in zip (data_full['lat'],data_full['long'],data_full['Active Cases'],data_full['state']):
    folium.CircleMarker([lat,lon],radius=value*0.3,popup=('<strong>State</strong>:'+str(name).capitalize() + '<br>''<strong>Active Cases</strong>:'+str(value)+'<br>'),color='blue',fill_color='blue',fill_opacity=0.3).add_to(map)
map

import seaborn as sns
f,ax=plt.subplots(figsize=(12,8))
data=data_full[['state','Confirmed','cured','deaths']]
data.sort_values('Confirmed',ascending=False,inplace=True)
sns.set_color_codes("pastel")
sns.barplot(x="Confirmed",y="state",data=data,label="Confirmed",color="r")
sns.set_color_codes("muted")
sns.barplot(x='cured',y='state',data=data,label="Recovered",color="g")
ax.legend(ncol=2,loc="lower right",frameon=True)
ax.set(xlim=(0,35),ylabel="",xlabel="cases")
sns.despine(left=True,bottom=True)
fig=go.Figure()
fig.add_trace(go.Scatter(x=dbd_India['Date'],y=dbd_India['Total Cases'],mode='lines+markers',name='Total Cases'))
fig.add_trace(go.Scatter(x=dbd_India['Date'],y=dbd_India['Recovered'],mode='lines',name='Recovered'))
fig.add_trace(go.Scatter(x=dbd_India['Date'],y=dbd_India['Active'],mode='lines',name='Active'))
fig.add_trace(go.Scatter(x=dbd_India['Date'],y=dbd_India['Deaths'],mode='lines',name='Deaths'))
fig.update_layout(title_text="Trend of Coronavirus Cases in India(Cumulative cases)",plot_bgcolor='rgb(250,242,242)')
fig.show()
fig=px.bar(dbd_India,x="Date",y="New Cases",barmode='group',height=400)
fig.update_layout(title_text='New Coronavirus Cases in India per day',plot_bgcolor='rgb(250,242,242)')
fig.show()

fig=px.bar(dbd_Italy,x="Date",y="Total Cases",color='Total Cases',orientation='v',height=600,title='Confirmed Cases in Italy',color_discrete_sequence=px.colors.cyclical.mygbm)
fig.update_layout(plot_bgcolor='rgb(250,242,242)') 
fig.show()
fig=px.bar(dbd_Korea,x="Date",y="Total Cases",color='Total Cases',orientation='v',height=600,title='Confirmed Cases in Korea',color_discrete_sequence=px.colors.cyclical.mygbm)
                                      
                                                                                            

fig.update_layout(plot_bgcolor='rgb(250,242,242)')
fig.show()
fig=px.bar(dbd_India,x="Date",y="Total Cases",color='Total Cases',orientation='v',height=600,title='Confirmed Cases in India',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.update_layout(plot_bgcolor='rgb(250,242,242)')
fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig=make_subplots(
    rows=2,cols=2,specs=[[{},{}],[{"colspan":2},None]],subplot_titles=("S.Korea","Italy","India"))
fig.add_trace(go.Bar(x=dbd_Korea['Date'],y=dbd_Korea['Total Cases'],marker=dict(color=dbd_Korea['Total Cases'],coloraxis="coloraxis")),1,1)
fig.add_trace(go.Bar(x=dbd_India['Date'],y=dbd_India['Total Cases'],marker=dict(color=dbd_India['Total Cases'],coloraxis="coloraxis")),1,2)
fig.add_trace(go.Bar(x=dbd_Italy['Date'],y=dbd_Italy['Total Cases'],marker=dict(color=dbd_Italy['Total Cases'],coloraxis="coloraxis")),2,1)
fig.update_layout(coloraxis=dict(colorscale="Bluered_r"),showlegend=False,title_text="Total Confirmed cases(Cumulative)")
fig.update_layout(plot_bgcolor='rgb(250,242,242)')
fig.show()


title='Main Source for News'
labels=['Korea','Italy','India']
colors=['rgb(0,128,0)','rgb(255,0,0)','rgb(49,130,189)']
mode_size=[8,8,12]
line_size=[2,2,4]
fig=go.Figure()
fig.add_trace(go.Scatter(x=dbd_Korea['Days after surpassing 100 cases'],y=dbd_Korea['Total Cases'],mode='lines',name=labels[0],line=dict(color=colors[0],width=line_size[0]),connectgaps=True))
fig.add_trace(go.Scatter(x=dbd_India['Days after surpassing 100 cases'],y=dbd_India['Total Cases'],mode='lines',name=labels[2],line=dict(color=colors[2],width=line_size[0]),connectgaps=True))
fig.add_trace(go.Scatter(x=dbd_Italy['Days after surpassing 100 cases'],y=dbd_Italy['Total Cases'],mode='lines',name=labels[1],line=dict(color=colors[1],width=line_size[0]),connectgaps=True))
annotations=[]
annotations.append(dict(xref='paper',yref='paper',x=0.5,y=-0.1,xanchor='center',yanchor='top',text='Days after surpassing 100 cases',font=dict(family='Arial',size=12,color='rgb(150,150,150)'),showarrow=False))
fig.update_layout(annotations=annotations,plot_bgcolor='white',yaxis_title='Cummulative cases')
fig.show()


fig=px.bar(data.sort_values('deaths',ascending=False).head(25).sort_values('deaths',ascending=True),x="deaths",y="state",title="Deaths",text='deaths',orientation='h',width=700,height=700,range_x=[0,max(data['deaths'])+500])
fig.update_traces(marker_color='#ff1e56',opacity=0.8,textposition='outside')
fig.show()
fig=px.bar(data.sort_values('cured',ascending=False).head(25).sort_values('cured',ascending=True),x="cured",y="state",title="cured",text='cured',orientation='h',width=700,height=700,range_x=[0,max(data['cured'])+500])
fig.update_traces(marker_color='#ff1e56',opacity=0.8,textposition='outside')
fig.show()
fig=px.bar(data.sort_values('Confirmed',ascending=False).head(25).sort_values('Confirmed',ascending=True),x="Confirmed",y="state",title="Confirmed",text='Confirmed',orientation='h',width=700,height=700,range_x=[0,max(data['Confirmed'])+500])
fig.update_traces(marker_color='#ff1e56',opacity=0.8,textposition='outside')
fig.show()
fig=px.bar(data_full.sort_values('Active Cases',ascending=False).head(25).sort_values('Active Cases',ascending=True),x="Active Cases",y="state",title="Active Cases",text='Active Cases',orientation='h',width=700,height=700,range_x=[0,max(data_full['Active Cases'])+500])
fig.update_traces(marker_color='#ff1e56',opacity=0.8,textposition='outside')
fig.show()

