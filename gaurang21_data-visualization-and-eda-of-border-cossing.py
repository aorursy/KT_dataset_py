import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')

df
df.shape
df.describe()
df.info()
df.head()
df["Border"] = ["Mexico" if i == "US-Mexico Border" else "Canada" for i in df.Border]
print("Unique port name: ",len(df["Port Name"].unique()))

print("Unique States: ",len(df["State"].unique()))

print("Unique Port Codes: ",len(df["Port Code"].unique()))

print("Unique Borders: ",len(df["Border"].unique()))

print("Unique measure of entries: ",len(df["Measure"].unique()))
df.isnull().any()
import folium

world_map =folium.Map(location=[37.09, -95.71],zoom_start=4)

world_map
#temp = df['Location']

df['Location']
df['Location'] = df['Location'].str.lstrip('POINT')

df['Location']
'''df['Location'] = df['Location'].map(lambda x: x.lstrip('('))

df['Location'].str.lstrip(')')

df['Location']'''

df['Location']= df['Location'].str.replace(r'\)', '')

df['Location']= df['Location'].str.replace(r'\(', '')

df['Location']
df['Location'].shape
temp = df['Location']

temp.to_frame()

temp[0]

temp.shape

temp = pd.DataFrame(df['Location'].str.split(' ',2).tolist(),columns = ['z','Y','X'])

temp

temp.X.shape
df.insert(8, "X", temp.X, True) 

df.insert(9,"Y",temp.Y,True)
df
df.head()
df =df.iloc[:1000,:]

df.shape

df['Port Name']
#Create entry_map for plotting

entry_map =folium.Map(location=[37.09, -95.71],zoom_start=4)

entry_map
entries = folium.map.FeatureGroup()

for lat,lon in zip(df.X,df.Y):

    entries.add_child(

    folium.CircleMarker(

    [lat,lon],

    radius=5,

    color='yellow',

    fill=True,

    fill_color='blue',

    fill_opacity=0.6,

        )

    )

entry_map.add_child(entries)
latitudes = list(df.X)

longitudes = list(df.Y)

labels = list(df['Port Name'])



for lat, lng, label in zip(latitudes, longitudes, labels):

    folium.Marker([lat, lng], popup=label).add_to(entry_map)    

    

# add incidents to map

entry_map.add_child(entries)
entry_map = folium.Map(location=[37.09, -95.71], zoom_start=4)



# loop through the 100 crimes and add each to the map

for lat, lng, label in zip(df.X, df.Y, df.Measure):

    folium.CircleMarker(

        [lat, lng],

        radius=5, # define how big you want the circle markers to be

        color='yellow',

        fill=True,

        popup=label,

        fill_color='blue',

        fill_opacity=0.6

    ).add_to(entry_map)



# show map

entry_map
from folium import plugins



cluster_map = folium.Map(location = [37.09, -95.71], zoom_start = 4)



entries = plugins.MarkerCluster().add_to(cluster_map)



# loop through the dataframe and add each data point to the mark cluster

for lat, lng, label, in zip(df.X, df.Y, df['Port Name']):

    folium.Marker(

        location=[lat, lng],

        icon=None,

        popup=label,

    ).add_to(entries)



# display map

cluster_map
plt.figure(figsize=(18,5))

ax = sns.violinplot(x="State", y="Value",

                    data=df[df.Value < 30000],

                    scale="width", palette="Set3")
df_state =df.groupby('State',axis=0).sum()

df_state=df_state.drop(columns=['Port Code'])

#Top 5 States

df_state = df_state.sort_values(['Value'], ascending=[0])

df_state

limit = 6

df_state = df_state.iloc[:limit,:]

df_state
explode_list = [0, 0, 0, 0.1,0.1,0.1]



df_state['Value'].plot(kind='pie',figsize=(15,6), autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,explode=explode_list,labels=None)

plt.title('Border crossed based on the States',y=1.12)

plt.legend(labels=df_state.index, loc='upper left')

plt.axis('equal') 

plt.show()
df_measure=df.groupby('Measure',axis=0).sum()

df_measure=df_measure.drop(columns=['Port Code'])

df_measure=df_measure.sort_values(['Value'],ascending=[0])

limit=6

df_measure=df_measure.iloc[:limit,:]

df_measure
explode_list = [0, 0,0,0.1,0.1,0.1]



df_measure['Value'].plot(kind='pie',figsize=(15,6), autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,explode=explode_list,labels=None)

plt.title('Border crossed based on measures of crossing',y=1.12)

plt.legend(labels=df_measure.index, loc='upper left')

plt.axis('equal') 

plt.show()
df_country=df.groupby('Border').sum()

df_country=df_country.drop(columns=['Port Code'])

df_country
df_country['Value'].plot(kind='pie',figsize=(15,6),autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,labels=None)

plt.title('Border crossed based on country',y=1.12)

plt.legend(labels=df_country.index, loc='upper left')

plt.axis('equal') 

plt.show()
df_port=df.groupby('Port Name').sum()

df_port=df_port.drop(columns=['Port Code'])

df_port=df_port.sort_values(['Value'],ascending=[0])

df_port.head(10)

df_port = df_port.iloc[:10,:]

df_port
exp_list = [0, 0,0,0,0,0,0.1,0.1,0.1,0.1]



df_port['Value'].plot(kind='pie',figsize=(15,6), autopct='%1.1f%%',startangle=90,shadow=True,pctdistance=1.12,explode=exp_list,labels=None)

plt.title('Border crossed based on measures of crossing',y=1.12)

plt.legend(labels=df_port.index, loc='upper left')

plt.axis('equal') 

plt.show()
df
df_state =df.groupby('State',axis=0).sum()

df_state=df_state.drop(columns=['Port Code'])

#Top 5 States

df_state = df_state.sort_values(['Value'], ascending=[0])

df_state
df_state.plot(kind='bar', figsize=(10, 6), rot=90) 
df_measure=df.groupby('Measure',axis=0).sum()

df_measure=df_measure.drop(columns=['Port Code'])

df_measure=df_measure.sort_values(['Value'],ascending=[0])

df_measure
df_measure.plot(kind='bar', figsize=(10, 6), rot=90) 
df_country.plot(kind='bar',figsize=(10,6),rot=90)
df_port.plot(kind='bar',figsize=(10,6),rot=90)
#Using Seaborn

import seaborn as sns



plt.figure(figsize=(18,7))

sns.barplot(x = df_state.index, y = "Value", data = df_state)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(18,7))

sns.barplot(x = "State", y = "Value", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(18,7))

sns.barplot(x = "State", y = "Value",hue='Border', data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(18,7))

sns.barplot(x = df_measure.index, y = "Value", data = df_measure)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(18,7))

sns.barplot(x = "Measure", y = "Value", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(18,7))

sns.barplot(x = df_port.index, y = "Value", data = df_port)

plt.xticks(rotation=45)

plt.show()
sns.boxenplot(x="Border", y="Value",

              color="b",

              scale="linear", data=df)

plt.show()
df
from wordcloud import WordCloud, ImageColorGenerator

text=" ".join(str(port) for port in df['Port Name'])

text
word_cloud = WordCloud(max_words=100,background_color='white').generate(text)

plt.figure(figsize=(15,10))

plt.imshow(word_cloud, interpolation='bilinear')

plt.axis("off")

plt.show()
fig,ax = plt.subplots(1,2,figsize=(15,5))

chart1=sns.countplot(df['Measure'],hue='Border',data=df,ax=ax[1])

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=90)

chart2=sns.countplot(df['Measure'],data=df,ax=ax[0])

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=90)

ax[0].title.set_text("Measure:data")

ax[1].title.set_text("Measure:Border")
fig,ax = plt.subplots(1,2,figsize=(15,5))

chart1=sns.countplot(df['State'],hue='Border',data=df,ax=ax[1])

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=45)

chart2=sns.countplot(df['State'],data=df,ax=ax[0])

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=90)

ax[0].title.set_text("State:data")

ax[1].title.set_text("State:Border")
df_date = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')

df_date
df_date["DateAsDateObj"] = pd.to_datetime(df_date.Date)

df_date = df_date.set_index("DateAsDateObj")

df_date
dataForPlot = df_date.resample("M").mean()

dataForPlot.loc[:,["Value"]].plot()
dataForPlot = df_date.resample("M").mean()

dataForPlot.loc[:,["Value"]].plot()
##Plot  by year

dataForPlot = df_date.loc[:,["Measure","Value"]]

dataForPlot = dataForPlot.groupby("Measure").resample("Y").mean()

dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="Measure", values="Value").plot(subplots=True, figsize=(8,14))
dataForPlot = df_date.loc[:,["State","Value"]]

dataForPlot = dataForPlot.groupby("State").resample("Y").mean()

dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="State", values="Value").plot(subplots=True, figsize=(8,14))
dataForPlot = df_date.loc[:,["Measure","Value"]]

dataForPlot = dataForPlot.groupby("Measure").resample("M").mean()

dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="Measure", values="Value").plot(subplots=True, figsize=(8,14))
dataForPlot = df_date.loc[:,["State","Value"]]

dataForPlot = dataForPlot.groupby("State").resample("M").mean()

dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="State", values="Value").plot(subplots=True, figsize=(8,14))