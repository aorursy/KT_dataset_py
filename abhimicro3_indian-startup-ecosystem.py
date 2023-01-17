import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
color=sns.color_palette()
%matplotlib inline

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster

import squarify
from subprocess import check_output
print(check_output(['ls','../input']).decode('utf8'))

import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/startup_funding.csv')

df.head()
df.columns
df.dtypes
total = df.isnull().sum().sort_values(ascending = False).reset_index()
total.columns=['Features','Total%']
total['Total%']=(total['Total%']*100)/df.shape[0]
total
df['Date'][df['Date']=='12/05.2015'] = '12/05/2015'
df['Date'][df['Date']=='13/04.2015'] = '13/04/2015'
df['Date'][df['Date']=='15/01.2015'] = '15/01/2015'
df['Date'][df['Date']=='22/01//2015'] = '22/01/2015'
df['Date']=pd.to_datetime(df['Date'],format='%d/%m/%Y')
df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month
def num_invest(x):
    investors=x.split(',')
    return len(investors)

df.dropna(how='any',axis=0,subset=['InvestorsName'],inplace=True)
df['Num_Investors']=df['InvestorsName'].apply(lambda x:num_invest(x))
print(df.head())
df.drop(['Remarks','Date'],axis=1,inplace=True)
df['AmountInUSD']=df['AmountInUSD'].apply(lambda x:float(str(x).replace(',','')))
df['AmountInUSD']=pd.to_numeric(df['AmountInUSD'])
startup=df['StartupName'].value_counts().sort_values(ascending=False).reset_index().head(10)
startup.columns=['Startup','Count']
data = [go.Bar(
            x=startup.Startup,
            y=startup.Count,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')
top=df.groupby(['StartupName']).sum().sort_values(by='AmountInUSD',ascending=False).reset_index().head(10)
data = [go.Bar(
            x=top.StartupName,
            y=top.AmountInUSD,
             opacity=0.6
    )]

py.iplot(data, filename='basic-bar')
city=df.CityLocation.value_counts().sort_values(ascending=False).reset_index().head(10)
city.columns=['City','Total']
fig,ax=plt.subplots(figsize=(15,10))
sns.barplot('City','Total',data=city,ax=ax)
plt.xlabel('City',fontsize=10)
plt.ylabel('StartUp Getting Funds',fontsize=10)
plt.title('Funding Across Cities',fontsize=15)
plt.show()
city_loc=pd.DataFrame({'City':['Bangalore','Mumbai','Chennai','New Delhi','Gurgaon','Pune','Noida','Hyderabad','Ahmedabad','Jaipur','Kolkata'],
                      'X':[12.9716,19.0760,13.0827,28.7041,28.4595,18.5204,28.5355,17.3850,23.0225,26.9124,22.5726],
                      'Y':[77.5946,72.8777,80.2707,77.1025,77.0266,73.8567,77.3910,78.4867,72.5714,75.7873,88.3639]})
city_loc

m = folium.Map(
    location=[20.5937,78.9629],
    tiles='Cartodb Positron',
    zoom_start=5
)

marker_cluster = MarkerCluster(
    name='City For StartUps',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(city_loc.shape[0]):
    location = city_loc.X.values[k], city_loc.Y.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = city_loc.City.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")

m
tech_bang=df[df['CityLocation']=='Bangalore']
tech_bang=tech_bang['IndustryVertical'].value_counts().sort_values(ascending=False).head(5).reset_index()
tech_bang.columns=['Industry','Count']
tech_bang
tech_mum=df[df['CityLocation']=='Mumbai']
tech_mum=tech_mum['IndustryVertical'].value_counts().sort_values(ascending=False).head(5).reset_index()
tech_mum.columns=['Industry','Count']
tech_new=df[df['CityLocation']=='New Delhi']
tech_new=tech_new['IndustryVertical'].value_counts().sort_values(ascending=False).head(5).reset_index()
tech_new.columns=['Industry','Count']

fig = {
  "data": [
    {
      "values": np.array((tech_bang['Count'] / tech_bang['Count'].sum())*100),
      "labels": tech_bang.Industry,
      "domain": {"x": [0, 0.48],'y':[0,0.48]},
      "name": "Bengalurelore",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },     
    {
      "values": np.array((tech_mum['Count'] / tech_mum['Count'].sum())*100),
      "labels": tech_mum.Industry,
      "text":"Mumbai",
      "textposition":"inside",
      "domain": {"x": [.52, 1],"y":[0,.48]},
      "name": "CO2 Emissions",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },
   {
      "values": np.array((tech_new['Count'] / tech_new['Count'].sum())*100),
      "labels": tech_new.Industry,
      "domain": {"x": [0.1, .88],'y':[0.52,1]},
      "name": "New Delhi",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    }],
  "layout": {
        "title":"Donut Chart For Industry in Top 3 Cities",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Bengaluru",
                "x": 0.15,
                "y": -0.08
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Mumbai",
                "x": 0.8,
                "y": -0.08
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "New Delhi",
                "x": 0.20,
                "y": 0.8
            }
        ]
    }
}
py.iplot(fig, filename='donut')
vertical=df['IndustryVertical'].value_counts().sort_values(ascending=False).reset_index().head(10)
vertical.columns=['vertical','Count']
tag = (np.array(vertical.vertical))
sizes = (np.array((vertical['Count'] / vertical['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top Industry To get Frequent Fund')
data = [trace]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Activity Distribution")
vertical=df['SubVertical'].value_counts().sort_values(ascending=False).reset_index().head(10)
vertical.columns=['vertical','Count']
tag = (np.array(vertical.vertical))
sizes = (np.array((vertical['Count'] / vertical['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top SubVerticals To get Frequent Fund')
data = [trace]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Activity Distribution")
df['YearMonth']=df['Year']*100+df['Month']
val=df['IndustryVertical'].value_counts().reset_index().head(10)
val.columns=['Industry','Count']
x=val.Industry
data=[]
for i in x:
    industry=df[df['IndustryVertical']==i]
    year_count=industry['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')
df['InvestmentType'][df['InvestmentType']=='SeedFunding']='Seed Funding'
df['InvestmentType'][df['InvestmentType']=='PrivateEquity']='Private Equity'
df['InvestmentType'][df['InvestmentType']=='Crowd funding']='Crowd Funding'

val=df['InvestmentType'].value_counts().reset_index().head(10)
val.columns=['Investment','Count']
x=val.Investment
data=[]
for i in x:
    industry=df[df['InvestmentType']==i]
    year_count=industry['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')
total_invs=df.groupby('Month').sum().reset_index()
total_invs
trace = go.Scatter(
    x = total_invs.Month,
    y = total_invs.AmountInUSD
)

data = [trace]

py.iplot(data, filename='basic-line')
val=df['CityLocation'].value_counts().reset_index().head(10)
val.columns=['City','Count']
x=val.City
data=[]
for i in x:
    industry=df[df['CityLocation']==i]
    year_count=industry['Month'].value_counts().reset_index().sort_values(by='index')
    year_count.columns=['Month','Count']
    trace = go.Scatter(
    x = year_count.Month,
    y = year_count.Count,
    name = i)
    data.append(trace)
    

py.iplot(data, filename='basic-line')
from wordcloud import WordCloud

names = df["InvestorsName"][~pd.isnull(df["InvestorsName"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Investor Names", fontsize=35)
plt.axis("off")
plt.show()
df['InvestorsName'][df['InvestorsName'] == 'Undisclosed investors'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'undisclosed Investors'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'undisclosed investors'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'Undisclosed investor'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'Undisclosed Investor'] = 'Undisclosed Investors'
df['InvestorsName'][df['InvestorsName'] == 'Undisclosed'] = 'Undisclosed Investors'
investors = df['InvestorsName'].value_counts().head(10)
print(investors)
plt.figure(figsize=(15,8))
sns.barplot(investors.index, investors.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Investors Names', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Investors Names with number of funding", fontsize=16)
plt.show()
investors = df['Num_Investors'].value_counts().head(10)
print(investors)
plt.figure(figsize=(15,8))
sns.barplot(investors.index, investors.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Number Of Investors', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Number Of Investors Per Funded Project", fontsize=16)
plt.show()