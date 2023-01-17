#import libraries



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt 

import seaborn as sns 



import plotly.tools as tls

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



import plotly.express as px
#read dataset

noise_complaints_data = pd.read_csv('../input/Noise_Complaints.csv')
#preview data

noise_complaints_data.head()
#show basic information of the dataset

noise_complaints_data.info()
#count missing items

noise_complaints_data.isna().sum()
#plot null values on a heatmap



sns.heatmap(noise_complaints_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#drop unneccessary columns

noise_complaints_data= noise_complaints_data.drop(['Status','Due Date','Agency','Agency Name','Landmark','Facility Type','Status','Due Date',

            'Resolution Description','Community Board','Park Facility Name','Park Borough','Vehicle Type',

            'Taxi Company Borough','Taxi Pick Up Location','Bridge Highway Name','Bridge Highway Direction','Road Ramp','Bridge Highway Segment'],axis=1)
#count complaints by description

comp_desc = (noise_complaints_data['Descriptor'].value_counts())

comp_desc
fig = go.Figure(data=[go.Histogram(y=noise_complaints_data['Descriptor'])])



fig.update_layout(

    autosize=False,

    width=800,

    height=800, 

    title=go.layout.Title(text="NYC Noise Complaint Descriptors 2018"))
#view count of different kinds of complaints reported

comp_type = (noise_complaints_data['Complaint Type'].value_counts())

comp_type
sns.countplot(y='Complaint Type',

              data=noise_complaints_data,

              order= noise_complaints_data['Complaint Type'].value_counts().index).set_title('NYC Noise Complaint Types 2018')
#find unique borough names

noise_complaints_data['Borough'].unique()
#group count of complaints by borough

count_borough = noise_complaints_data['Borough'].value_counts()

count_borough
#create pie chart

fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(aspect='equal'))



borough = list(count_borough.index)

count = list(count_borough)



def func(pct):

    absolute = int(pct/100.*np.sum(count))

    return "{:.1f}% ({:d})".format(pct, absolute)





ax.pie(count_borough, 

       autopct=lambda pct: func(pct),

      pctdistance=1.2)



ax.legend(borough,

          title="Borough",

          loc="center left",

          bbox_to_anchor=(1, 0.3, 0.5, 1))



plt.title('NYC Noise Complaints by Borough 2018')
noise_complaints_data['City'].unique()
comp_city = (noise_complaints_data['City'].value_counts())
comp_city_df = comp_city.to_frame().reset_index()

comp_city_df.rename(columns={'index':'City Name'},inplace=True)

comp_city_df.rename(columns={'City':'Count'},inplace=True)

comp_city_df.head(10)
sns.barplot(x='Count',

            y='City Name',

            data=comp_city_df.head(10)).set_title('Top 10 NYC Noise Compaints by City 2018')
noise_complaints_data['Location Type'].value_counts()
sns.countplot(y='Location Type',

              data=noise_complaints_data,

              order= noise_complaints_data['Location Type'].value_counts().index).set_title('NYC Noise Complaints by Location Type 2018')
#find maximum and minimum longitude

print(noise_complaints_data['Longitude'].max())

print(noise_complaints_data['Longitude'].min())
#find maximum and minimum latitude

print(noise_complaints_data['Latitude'].max())

print(noise_complaints_data['Latitude'].min())
#plot complaints



#resize plot

from pylab import rcParams

rcParams['figure.figsize'] = 10, 10



plt.plot(noise_complaints_data['Longitude'],noise_complaints_data['Latitude'],'.',markersize=0.2)

plt.title('NYC Noise Complaints 2018')
#closer look at Manhattan, Brooklyn, Bronx, Queens



#resize plot

from pylab import rcParams

rcParams['figure.figsize'] = 30, 13



#create subplot 

fig, axes = plt.subplots(nrows=1,ncols=4)



#filter boroughs

noise_complaints_manhattan= noise_complaints_data[noise_complaints_data['Borough']=='MANHATTAN']

noise_complaints_brooklyn= noise_complaints_data[noise_complaints_data['Borough']=='BROOKLYN']

noise_complaints_bronx= noise_complaints_data[noise_complaints_data['Borough']=='BRONX']

noise_complaints_queens= noise_complaints_data[noise_complaints_data['Borough']=='QUEENS']



#plot

axes[0].plot(noise_complaints_manhattan['Longitude'],noise_complaints_manhattan['Latitude'],'.',markersize=0.6)

axes[0].set_title('Manhattan Noise Complaints 2018')



axes[1].plot(noise_complaints_brooklyn['Longitude'],noise_complaints_brooklyn['Latitude'],'.',markersize=0.6)

axes[1].set_title('Brooklyn Noise Complaints 2018')



axes[2].plot(noise_complaints_bronx['Longitude'],noise_complaints_bronx['Latitude'],'.',markersize=0.6)

axes[2].set_title('Bronx Noise Complaints 2018')



axes[3].plot(noise_complaints_queens['Longitude'],noise_complaints_queens['Latitude'],'.',markersize=0.6)

axes[3].set_title('Queens Noise Complaints 2018')

#show geographical heatmap; show only 30000 datpoints due to memory limit



head = noise_complaints_data.head(30000)



import folium

from folium.plugins import HeatMap

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(head[['Latitude','Longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
#find maximum and minimum longitude

print(noise_complaints_data['Latitude'].mean())

print(noise_complaints_data['Longitude'].mean())
#check the current created date format

type(noise_complaints_data['Created Date'].iloc[0])
#convert created date to date time format

noise_complaints_data['Created Date'] = pd.to_datetime(noise_complaints_data['Created Date'])
#create hour, month, day of week columns

noise_complaints_data['Hour'] = noise_complaints_data['Created Date'].apply(lambda time: time.hour)

noise_complaints_data['Month'] = noise_complaints_data['Created Date'].apply(lambda time: time.month)

noise_complaints_data['Day of Week'] = noise_complaints_data['Created Date'].apply(lambda time: time.dayofweek)
#check format of the day of the week column

noise_complaints_data['Day of Week'].head()
#convert day of the week from integer to string descriptions

d_o_w = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

noise_complaints_data['Day of Week'] = noise_complaints_data['Day of Week'].map(d_o_w)
#plot count of complaints by time



#resize plot

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5



sns.countplot(x=noise_complaints_data['Hour'])

plt.title('NYC Count of Complaints 2018: by Hour')
#plot count of complaints by day of week



sns.countplot(x=noise_complaints_data['Day of Week'])

plt.title('NYC Count of Complaints 2018: by Day of Week')
#plot count of complaints by month



sns.countplot(x=noise_complaints_data['Month'])

plt.title('NYC Count of Complaints 2018: by Month')
noise_complaints_data['Date'] = noise_complaints_data['Created Date'].apply(lambda t:t.date())
noise_complaints_data.groupby(noise_complaints_data['Date']).count()['Unique Key'].plot()

plt.tight_layout()

plt.title('NYC Count of Complaints 2018: by Date')
#create data frame of count of complaints on day of week vs hour

dayHour = noise_complaints_data.groupby(by=['Day of Week','Hour']).count()['Unique Key'].unstack()

dayHour
#create heatmap of count of complaints on day of week vs hour

sns.heatmap(data=dayHour,cmap='coolwarm')

plt.title('NYC Count of Complaints 2018: Day of Week vs. Hour')
#create data frame of count of complaints on month vs hour

monthDay = noise_complaints_data.groupby(by=['Month','Day of Week']).count()['Unique Key'].unstack()

monthDay
#create heatmap of count of complaints on month vs hour

sns.heatmap(data=monthDay,cmap='coolwarm')

plt.title('NYC Count of Complaints 2018: Day of Week vs. Month')