#Realizing Imports

%matplotlib inline
import pandas as pd
import numpy as np
import calendar
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.plotly as py
import plotly.graph_objs as go

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from folium.plugins import HeatMap
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

sns.set_style('darkgrid')
init_notebook_mode(connected=True)
#Reading Dataset

df = pd.read_csv('../input/911.csv', ',')
#Creating variable Reason

df['Type'] = df['title'].apply(lambda s:s.split(':')[0])
df['Reason'] = df['title'].apply(lambda s:s.split(':')[1])
#Classification of Variables

table = [["Type","Qualitative Nominal"],["Reason","Qualitative Nominal"],["Week","Qualitative Nominal"],
            ["Week_Abbr","Qualitative Nominal"],["Month","Discrete Quantitative"],["Month_Abbr","Discrete Quantitative"],
            ["Year","Discrete Quantitative"],["Hour","Discrete Quantitative"],["timeStamp","Qualitative Ordinal"],
            ["Day","Qualitative Ordinal"],["Date","Discrete Quantitative"],["day/night","Qualitative Nominal"]]

filing = pd.DataFrame(table, columns=["Variable", "Classification"])
filing
#Creating lists with the Names of the Week and the Months

wday = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
months = ['','January','February','March','April','May','June','July','August','September','October','November','December']
#Converting to datetime and creating hour / hour, day / day, month / month, year / year, week / week

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Day'] = df['timeStamp'].apply(lambda x: x.day)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Month_Abbr'] = df['timeStamp'].apply(lambda x: months[x.month])
df['Year'] = df['timeStamp'].apply(lambda x: x.year)
df['Week'] = df['timeStamp'].apply(lambda x: x.weekday())
df['Week_Abbr'] = df['timeStamp'].apply(lambda x: wday[x.weekday()])
df['Date']=df['timeStamp'].apply(lambda x:x.date())
df["day/night"] = df["timeStamp"].apply(lambda x : "night" if int(x.strftime("%H")) > 19 else "day")
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "Year", data= df[df['Year'] == 2016], palette="viridis", hue = "Type")
plt.title("Records of the Events of 2016")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "Month_Abbr", data= df[df['Year'] == 2016], palette="viridis", hue= "Type")
plt.title("Monthly records of occurrences of 2016")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set_context("paper", font_scale = 2)
sns.countplot(x= "Week_Abbr", data= df[df['Year'] == 2016], palette="viridis", hue= "Type" )     
plt.title("Weekly records of the occurrences of 2016")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
def timeZone(timestamp):
    hour = timestamp.hour
    if (hour > 6 and hour < 12) or hour == 6:
        return 'Morning'
    elif hour == 12:
        return 'Noon'
    elif hour > 12 and hour < 17:
        return 'Afternoon'
    elif (hour > 17 and hour < 21) or hour == 17:
        return 'Evening'
    elif (hour > 21 and hour < 6) or hour == 21:
        return 'Night'
df['timezone'] = df['timeStamp'].apply(lambda x : timeZone(x))  
sns.countplot('timezone', data = df[df['Year'] == 2016],palette="viridis", hue= "Type" )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Occurrences Records by Schedules")
plt.xticks(rotation=45)
df[(df['Type']=='EMS') & (df['Year'] == 2016)].groupby('Date').count()['twp'].plot(figsize=(15,3),label='EMS')
df[(df['Type']=='Fire') & (df['Year'] == 2016)].groupby('Date').count()['twp'].plot(figsize=(15,3),label='Fire')
df[(df['Type']=='Traffic') & (df['Year'] == 2016)].groupby('Date').count()['twp'].plot(figsize=(15,3),label='Traffic')
plt.title("Let's check the distribution of the different reasons")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
v0 = df[(df['title']=='Traffic: VEHICLE ACCIDENT -')].Hour.values
data = [go.Histogram(x=v0,histnorm='probability')]

layout = dict(title='Traffic: VEHICLE ACCIDENT (hr)',
            autosize= True,bargap= 0.015,height= 400,width= 500,hovermode= 'x',xaxis=dict(autorange= True,zeroline= False),
            yaxis= dict(autorange= True,showticklabels= True,))

fig1 = dict(data=data, layout=layout)
iplot(fig1)
v0 = df['timezone'].values
data = [go.Histogram(x=v0,histnorm='probability')]

layout = dict(title='Traffic: VEHICLE ACCIDENT (hr)',
            autosize= True,bargap= 0.015,height= 400,width= 500,hovermode= 'x',xaxis=dict(autorange= True,zeroline= False),
            yaxis= dict(autorange= True,showticklabels= True,))

fig1 = dict(data=data, layout=layout)
iplot(fig1)
W1 = df[(df['Type'] == 'Traffic') & (df['Week'] == 0)].groupby(['Hour']).size().reset_index(name='Qty')
W2 = df[(df['Type'] == 'Traffic') & (df['Week'] == 1)].groupby(['Hour']).size().reset_index(name='Qty')
W3 = df[(df['Type'] == 'Traffic') & (df['Week'] == 2)].groupby(['Hour']).size().reset_index(name='Qty')
W4 = df[(df['Type'] == 'Traffic') & (df['Week'] == 3)].groupby(['Hour']).size().reset_index(name='Qty')
W5 = df[(df['Type'] == 'Traffic') & (df['Week'] == 4)].groupby(['Hour']).size().reset_index(name='Qty')
W6 = df[(df['Type'] == 'Traffic') & (df['Week'] == 5)].groupby(['Hour']).size().reset_index(name='Qty')
W7 = df[(df['Type'] == 'Traffic') & (df['Week'] == 6)].groupby(['Hour']).size().reset_index(name='Qty')
# Dispersion
fig, ax = plt.subplots(figsize=(16, 10))

# Set up the plot
ax = plt.subplot(2, 2, 1)

ax.scatter(W1['Hour'], W1['Qty'], label="Monday")
ax.scatter(W2['Hour'], W2['Qty'], label="Tuesday")
ax.scatter(W3['Hour'], W3['Qty'], label="Wednesday")
ax.scatter(W4['Hour'], W4['Qty'], label="Thursday")
ax.scatter(W5['Hour'], W5['Qty'], label="Friday")
ax.scatter(W6['Hour'], W6['Qty'], label="Saturday")
ax.scatter(W7['Hour'], W7['Qty'], label="Sunday")
ax.legend(loc='best')
ax.legend(bbox_to_anchor=(1.25, 1.0))
ax.set_xlabel('Timestamp (hour)', fontsize=16,)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('Distribution Day of Week and Types Reason: Traffic', fontsize=12, fontweight='bold')

fig.tight_layout(pad=1.5, w_pad=2, h_pad=2.0)
top_10_twp=pd.DataFrame(df[(df['Type']=='Traffic') & (df['Year'] == 2016)]['twp'].value_counts().head(10))
top_10_twp.reset_index(inplace=True)
top_10_twp.columns=['Township','Count']
top_10_twp
top_10_twp=pd.DataFrame(df[(df['Type']=='Traffic') & (df['Year'] == 2016)]['twp'].value_counts().head(10))
top_10_twp.reset_index(inplace=True)
top_10_twp.columns=['Township','Count']
fig2=plt.figure(figsize=(12,6))
g=sns.barplot(data=top_10_twp,x='Township',y='Count',palette="viridis")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
fig2.tight_layout()
df[(df['Type']=='Traffic') & (df['Year'] == 2016) & (df['Day'] == 23) & (df['Month'] == 1)].groupby('Date').count()['twp']
df[(df['Type']=='Traffic') & (df['Year'] == 2016) & (df['Day'] == 23) & (df['Month'] == 1)].groupby('Date')['twp'].value_counts().head(20)
g = df[(df.Type =='Traffic') & (df['Year'] == 2016) & (df['Day'] == 23) & (df['Month'] == 1)]
p=pd.pivot_table(g, values='e', index=['Month_Abbr'] , columns=['Hour'], aggfunc=np.sum)
p.head()
cmap = sns.cubehelix_palette(light=2, as_cmap=True)
ax = sns.heatmap(p,cmap = cmap)
ax.set_title('Vehicle  Accidents - 23 January 2016 ');
df_lat_lng=df[(df.Type =='Traffic') & (df['Year'] == 2016) & (df['Day'] == 23) & (df['Month'] == 1)].groupby(['lat','lng'])['lat'].count()
df_lat_lng=df_lat_lng.to_frame()
df_lat_lng.columns.values[0]='count1'
df_lat_lng=df_lat_lng.reset_index()
lats=df_lat_lng[['lat','lng','count1']].values.tolist()
hmap = folium.Map(location=[40.4, -75.2], zoom_start=9, )
hmap.add_child(HeatMap(lats, radius = 5))
hmap
pd.show_versions ()
