import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS 
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import math

pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',1000)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/us-accidents/US_Accidents_June20.csv') #,nrows=10000)
redunCol=['ID','Source','End_Lat','End_Lng','Number','Street','Zipcode','Country','Weather_Timestamp'
            ,'Airport_Code','Astronomical_Twilight','Civil_Twilight','Nautical_Twilight']
df=df.drop(redunCol,axis=1)
df
words=''
for item in df['Description']:
    words+=' ' +str(item)
stopwords = set(STOPWORDS)
words=words.upper()
wordcloud = WordCloud(max_words=2000, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size =5).generate(words) 
# plot the WordCloud image                        
plt.figure(figsize = (10, 10), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()
stacked_data=[]
for i in range(4):
    temp=[0,0,0,0]
    stacked_data.append(temp)
    
for i in df.index:
    if df.loc[i,'Severity']>=1 and df.loc[i,'Severity']<=4:
        if str(df.loc[i,'Traffic_Calming'])=='False' and str(df.loc[i,'Traffic_Signal'])=='True':
            stacked_data[0][df.loc[i,'Severity']-1]+=1
        elif str(df.loc[i,'Traffic_Calming'])=='True' and str(df.loc[i,'Traffic_Signal'])=='False':
            stacked_data[1][df.loc[i,'Severity']-1]+=1
        elif str(df.loc[i,'Traffic_Calming'])=='True' and str(df.loc[i,'Traffic_Signal'])=='True':
            stacked_data[2][df.loc[i,'Severity']-1]+=1
        else:
            stacked_data[3][df.loc[i,'Severity']-1]+=1

x=['1','2','3','4']
plt.bar(x,stacked_data[0],0.4,label="traffic calming was not there but there was traffic signal")
plt.bar(x,stacked_data[1],0.4,bottom=stacked_data[0],label="traffic calming was there but no traffic signal")
bottom_1=list(np.add(stacked_data[0],stacked_data[1]))
plt.bar(x,stacked_data[2],0.4,bottom=bottom_1,label="traffic calming was there and there was traffic signal")
bottom_2=list(np.add(bottom_1,stacked_data[2]))
plt.bar(x,stacked_data[3],0.4,bottom=bottom_2,label="neither of them were present")
plt.rcParams["figure.figsize"] = (15, 8)
plt.title("Number of Accidents vs Severity number")
plt.xlabel("Severity number")
plt.ylabel("Number of accidents")
plt.legend()
plt.show()
state_count_acc = pd.value_counts(df['State'])

fig = go.Figure(data=go.Choropleth(
    locations=state_count_acc.index,
    z = state_count_acc.values.astype(float),
    locationmode = 'USA-states',
    colorscale = 'Reds',
    colorbar_title = "Count Accidents",
))

fig.update_layout(
    title_text = '2016 - 2019 US Traffic Accident Dataset by State',
    geo_scope='usa',
)

fig.show()
df1 = df.sample(n=10000)
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
cities = {}
state = []
lat_1 = []
lon_1 = []
for ind in df1.index:
    city = df1['City'][ind]
    state_1 = df1['State'][ind]
    if city in cities:
        cities[city] +=1
    else:
        cities[city] = 1
        state.append(state_1)
        lat_1.append(df1['Start_Lat'][ind])
        lon_1.append(df1['Start_Lng'][ind])

city_count = []
city_name = []
for city in cities:
    city_count.append(cities[city])
    city_name.append(city)
 
data1 = {'City':city_name}
data = pd.DataFrame(data1)  
data['State']= state
data['Accident_count'] = city_count
data['Longitude'] = lon_1
data['Latitude'] = lat_1
# Observe the result  
data
root = []
acc = []
for i in data.index:
    root.append(math.sqrt(data['Accident_count'][i]))
    acc.append(str(data['City'][i])+', Accidents: '+ str(data['Accident_count'][i]))
print(root,acc)
fig = go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = data['Longitude'],
        lat = data['Latitude'],
        text = acc,
        mode = 'markers',
        marker = dict(
            size = root,#data['Accident_count']/sum(data['Accident_count']),
            opacity = 0.8,
            reversescale = True,
            autocolorscale = True,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(255,0,0)'
            ),
           
        )))

fig.update_layout(
        title = 'Accidents in Cities',
        geo = dict(
            scope='usa',
            projection_type='albers usa',
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.7,
            subunitwidth = 0.7
        ),
    )
fig.show()

data_sever = df[['Start_Lng','Start_Lat','City','Visibility(mi)','Severity']].copy()
data_sever.dropna(inplace=True)

fig = go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = data_sever['Start_Lng'],
        lat = data_sever['Start_Lat'],
        text = data_sever['City'],
        mode = 'markers',
        marker = dict(
            size = data_sever['Visibility(mi)'],
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Blues',
            cmin = data_sever['Severity'].max(),
        color = data_sever['Severity'],
        cmax = 1,
            colorbar_title="Severity"
        )))

fig.update_layout(
        title = 'Severity & Visibility of accidents',
        geo = dict(
            scope='usa',
            projection_type='albers usa',
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.7,
            subunitwidth = 0.7
        ),
    )
fig.show()
data_sever = df.sample(n=10000)[['Start_Lng','Start_Lat','City','Visibility(mi)','Severity']]
data_sever.dropna(inplace=True)

fig = go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = data_sever['Start_Lng'],
        lat = data_sever['Start_Lat'],
        text = data_sever['City'],
        mode = 'markers',
        marker = dict(
            size = data_sever['Visibility(mi)'],
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Blues',
            cmin = data_sever['Severity'].max(),
        color = data_sever['Severity'],
        cmax = 1,
            colorbar_title="Severity"
        )))

fig.update_layout(
        title = 'Severity & Visibility of accidents',
        geo = dict(
            scope='usa',
            projection_type='albers usa',
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.7,
            subunitwidth = 0.7
        ),
    )
fig.show()
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time']=pd.to_datetime(df['End_Time'])
#getting duration
df['Duration']=df['End_Time']-df['Start_Time']
#adding duration in dataframe
#df.insert(6,"Duration",df['duration'])
#df.drop(['duration'],axis=1,inplace=True)
#converting duration to seconds
df['Duration'] = df['Duration'].dt.total_seconds()

#plotting scatter-plot
dfs=df.sample(n=10000)
df1=dfs.loc[dfs['Severity']==1]
df2=dfs.loc[dfs['Severity']==2]
df3=dfs.loc[dfs['Severity']==3]
df4=dfs.loc[dfs['Severity']==4]
x1=df1['Duration']
x1=np.array(x1)
y1=df1['Distance(mi)']
y1=np.array(y1)
x2=df2['Duration']
x2=np.array(x2)
y2=df2['Distance(mi)']
y2=np.array(y2)
x3=df3['Duration']
x3=np.array(x3)
y3=df3['Distance(mi)']
y3=np.array(y3)
x4=df4['Duration']
x4=np.array(x4)
y4=df4['Distance(mi)']
y4=np.array(y4)
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 20})
p1=plt.scatter(x1, y1, c='cyan')
p2=plt.scatter(x2, y2, c='red')
p3=plt.scatter(x3,y3,c='green')
p4=plt.scatter(x4,y4,c='blue')
plt.title('scatter plot of distance and severity vs duration')
plt.xlabel('duration in seconds')
plt.ylabel('distance')
plt.legend((p1,p2,p3,p4),
           ('severity 1','severity 2','severity 3','severity 4'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=10)
plt.show()
df['Start_Time']= pd.to_datetime(df['Start_Time'])
df['hour']= df['Start_Time'].dt.hour
df['year']= df['Start_Time'].dt.year
df['month']= df['Start_Time'].dt.month
df['week']= df['Start_Time'].dt.week
df['day']= df['Start_Time'].dt.day_name()
df['quarter']= df['Start_Time'].dt.quarter
df['time_zone']= df['Start_Time'].dt.tz
df['time']= df['Start_Time'].dt.time
plt.figure(figsize =(10,5))
df.groupby(['year']).size().sort_values(ascending=True).plot.bar()

plt.figure(figsize =(15,5))
df.groupby(['month']).size().plot.bar()
plt.figure(figsize =(15,5))
df.groupby(['year', 'month']).size().plot.bar()
plt.title('Number of accidents/year')
plt.ylabel('number of accidents')
plt.figure(figsize =(10,5))
df.groupby(['hour']).size().plot.bar()
plt.title('At which hour of day accidents happen')
plt.ylabel('count of accidents')
df['day_zone'] = pd.cut((df['hour']),bins=(0,6,12,18,24), labels=["night", "morning", "afternoon", "evening"])
plt.figure(figsize =(10,5))
df.groupby(['day_zone']).size().plot.bar()
df.groupby(['day']).size().plot.bar()
redunCol=['TMC','Start_Time','End_Time','Description','City','County','State','Timezone','Distance(mi)','Wind_Direction']
df=df.drop(redunCol,axis=1)
df.loc[df['Side']=='R','Side']=1 #Right side is 1
df.loc[df['Side']=='L','Side']=0 #Left side is 
df.loc[df['Sunrise_Sunset']=='Day','Sunrise_Sunset']=0 #daytime is 0
df.loc[df['Sunrise_Sunset']=='Night','Sunrise_Sunset']=1 #nighttime is 1
df