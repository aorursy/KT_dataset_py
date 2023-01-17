
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/traffic-collision-data-from-2010-to-present.csv')
df=df.drop(['DR Number','Date Reported','Address','Cross Street',
            'MO Codes','Crime Code Description'],axis=1)
df['year'] = pd.DatetimeIndex(df['Date Occurred']).year
df['month'] = pd.DatetimeIndex(df['Date Occurred']).month
df['day'] = pd.DatetimeIndex(df['Date Occurred']).day
df['time']=df['year']+(1/12)*df['month']+(1/360)*df['day']
df.isnull().sum()
df1=df.drop(['Victim Age'],axis=1)
df2=df1.drop(df1[df1['Victim Sex'].isnull()].index)
df2=df2.drop(df2[df2['Premise Description'].isnull()].index)
df2=df2.drop(df2[df2['Victim Descent'].isnull()].index)
df2.isnull().sum()
df2['logintude']=df2.Location.str[105:113]
df2['logintude']= df2['logintude'].str.replace("'", "", regex=True).str.replace(",", "", regex=True).str.replace(" ", "", regex=True).str.replace("l", "1", regex=True)
df2.loc[df2['logintude'].str[1:3]!= '11',['logintude']]='-118'
df2['logintude']=df2['logintude'].astype(float) 

df2['latitude']=df2.Location.str[130:136]
df2['latitude']= df2['latitude'].str.replace("'", "", regex=True).str.replace(",", "", regex=True).str.replace(" ", "", regex=True).str.replace("l", "1", regex=True).str.replace("}", "", regex=True)
#df2.loc[(df2['latitude'].str[0]== '.')|(df2['latitude'].str[0]==' ')|(df2['latitude'].str[0]=='0'),(df2['latitude'].str[0]=='n') ,['latitude']]=0
df2.loc[(df2['latitude'].str[0:2]!= '33') & (df2['latitude'].str[0:2]!= '34'),['latitude']]='32'
df2['latitude']=df2['latitude'].astype(float) 

df2=df2[df2['logintude']< -118.15]
df2=df2[df2['latitude']> 33.6]
df2=df2[df2['latitude']< 34.38]
df2.head()
df2['Victim Descent'].unique()
df2.loc[df2['Victim Descent']=='H',['Victim Descent']]='Hispanic'
df2.loc[df2['Victim Descent']=='B',['Victim Descent']]='Black'
df2.loc[df2['Victim Descent']=='O',['Victim Descent']]='Other'
df2.loc[df2['Victim Descent']=='W',['Victim Descent']]='White'
df2.loc[df2['Victim Descent']=='X',['Victim Descent']]='Unknown'
df2.loc[df2['Victim Descent']=='A',['Victim Descent']]='Asian'
df2.loc[df2['Victim Descent']=='K',['Victim Descent']]='Korean'
df2.loc[df2['Victim Descent']=='C',['Victim Descent']]='Chinese'
df2.loc[df2['Victim Descent']=='F',['Victim Descent']]='Filipino'
df2.loc[df2['Victim Descent']=='U',['Victim Descent']]='Hawaian'
df2.loc[df2['Victim Descent']=='J',['Victim Descent']]='Japaniese'
df2.loc[df2['Victim Descent']=='P',['Victim Descent']]='Pacific Islanders'
df2.loc[df2['Victim Descent']=='V',['Victim Descent']]='Vietnamise'
df2.loc[df2['Victim Descent']=='V',['Victim Descent']]='Vietnamise'
df2.loc[df2['Victim Descent']=='Z',['Victim Descent']]='Asian Indian'
df2.loc[df2['Victim Descent']=='I',['Victim Descent']]='American Indian'
df2.loc[df2['Victim Descent']=='G',['Victim Descent']]='Guamanian'
df2.loc[df2['Victim Descent']=='S',['Victim Descent']]='Samoan'
df2.loc[df2['Victim Descent']=='D',['Victim Descent']]='Cambodian'
df2.loc[df2['Victim Descent']=='L',['Victim Descent']]='Laotian'
df2.loc[df2['Victim Descent']=='-',['Victim Descent']]='Unknown'
df.time.hist()
df3=df2.sort_values(by=['time'])
df3.shape

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import numpy as np

x=df3.loc[:,'logintude'].values
y=df3.loc[:,'latitude'].values
xy=np.column_stack((x,y))
dt = 0.005
n=382600
L = 1
particles=np.zeros(n,dtype=[("position", float , 2),
                            ("velocity", float ,2),
                            ("force", float ,2),
                            ("size", float , 1)])

particles["position"]=(xy);
particles["velocity"]=np.zeros((n,2));
particles["size"]=0.5*np.ones(n);

fig = plt.figure(figsize=(7,7))
ax = plt.axes(xlim=(-119,-118),ylim=(33.5,34.4))
scatter=ax.scatter(particles["position"][:,0], particles["position"][:,1])

def update(frame_number):
   # particles["force"]=np.random.uniform(-2,2.,(n,2));
   # particles["velocity"] = particles["velocity"] + particles["force"]*dt
   # particles["position"] = particles["position"] + particles["velocity"]*dt

    #particles["position"] = particles["position"]%L
    scatter.set_offsets(xy[:100*frame_number,:])
    return scatter, 

anim = FuncAnimation(fig, update, interval=100)
from IPython.display import HTML
HTML(anim.to_jshtml())
mapbox_access_token='pk.eyJ1IjoiYW1tb24xIiwiYSI6ImNqbGhtdDNtNzFjNzQzd3J2aDFndDNmbmgifQ.-dt3pKGSvkBaSQ17qXVq3A'
df3=df2.sort_values('time')
df3=df2[df2['Victim Sex']=='F'].head(2000)
df4=df2[df2['Victim Sex']=='M'].head(2000)

data = [
    go.Scattermapbox(
        lat=df3['latitude'],
        lon=df3['logintude'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=0.3
        )),
    go.Scattermapbox(
        lat=df4['latitude'],
        lon=df4['logintude'],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.3
        ))]
layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=34.052363,
            lon=-117.960140
        ),
        pitch=0,
        zoom=7.8,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Crime')
plt.figure(figsize=(15,20))
ax=sns.scatterplot(x="logintude", y="latitude", hue="Victim Sex",data=df2)
plt.figure(figsize=(15,20))
ax=sns.scatterplot(x="logintude", y="latitude", hue="Victim Descent",data=df2)
plt.figure(figsize=(15,20))
df3=df2[(df2["Premise Description"]=='STREET')|(df2["Premise Description"]=='PARKING LOT')|(df2["Premise Description"]=='FREEWAY')|
        (df2["Premise Description"]=='SIDEWALK')]
ax=sns.scatterplot(x="logintude", y="latitude", hue="Premise Description",data=df3)
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['logintude'], df2['latitude'], df2['Time Occurred'], c='skyblue', s=60)
ax.view_init(30, 185)
plt.show()
sns.scatterplot(df2['logintude'],df2['Time Occurred'])
sns.scatterplot(df2['latitude'],df2['Time Occurred'])
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['logintude'], df2['latitude'], df2['Premise Code'], c='skyblue', s=60)
ax.view_init(30, 185)
plt.show()
df_premise=df.drop_duplicates('Premise Code',keep='first')
df_premise.sort_values('Premise Code')
df_premise_head=df_premise[['Premise Code','Premise Description']].copy()
df_premise_head.head(300)