import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

import folium
#Read the dataset

df = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv")
df.head()
#Check if there are any null values

print(df.isnull().sum())
#Function to get summary statistics for categorical variable.



def dataQuality(data):

    d={}

    def cat_quality(data):

        def count(x):

            return x.count()

        def miss_per(x):

            return x.isnull().sum()/len(x)

        def unique(x):

            return len(x.unique())

        def freq_cat(x):

            return x.value_counts().sort_values(ascending=False).index[0]

        def freq_cat_per(x):

            return x.value_counts().sort_values(ascending=False).index[0]/len(x)

        qr=dict()

        #select only categorical data types

        data=data.select_dtypes(include=[object])

        for i in np.arange(0,len(data.columns),1):

            xi=data.agg({data.columns[i]:[count,unique,miss_per,freq_cat]})

            qr[data.columns[i]]=xi.reset_index(drop=True)[data.columns[i]]

            df2=pd.DataFrame(qr)

            #df2.index=xi.index

        df2.index=["Count","Unique","Miss_percent","Freq_Level"]

        return df2.T

    d['categorical']=cat_quality(data)

    return d
#Call the above function to get the data quality report.

(dataQuality(df)['categorical'])
#Extract year from date and add to the dataframe 

df['year']=pd.DatetimeIndex(df['date']).year
plt.figure(figsize=(5,7))

splot=sns.countplot(data=df,x='year',palette='YlGnBu')

sns.set_style('ticks')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title("Year and number of shootings")

plt.xlabel('Year')

plt.ylabel('No. of deaths')

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(5,7))

splot=sns.countplot(data=df.query("armed == 'unarmed'").query("threat_level != 'attack'"),x='year',palette='YlGnBu')

sns.set_style('ticks')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title("Year and number of shootings")

plt.xlabel('Year')

plt.ylabel('No. of deaths')

plt.xticks(rotation=45)

plt.show()
#Drop all rows having null values

df=df.dropna(subset=['race'])



#Check if there are any null values

df.isnull().sum()
#Replce the acronyms with the actual words

def race(x):

    if(re.findall("W",x)):

        return 'White'

    elif(re.findall("B",x)):

        return 'Black'

    elif(re.findall("A",x)):

        return 'Asian'

    elif(re.findall("N",x)):

        return 'Native American'

    elif(re.findall("H",x)):

        return 'Hispanic'

    elif(re.findall("O",x)):

        return 'Other'

       

df['race']=df['race'].apply(lambda x:race(x))
plt.figure(figsize= (6,7))

splot= sns.countplot(data=df,x='race',palette='YlGnBu')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



sns.set_style('ticks')

plt.title("Race")

plt.xlabel('Race')

plt.ylabel('No. of deaths')

plt.xticks(rotation=45)

plt.show()
#Descriptive stats

df['age'].describe()
# Histogram to show the distribution

df['age'].plot.hist(grid=True, bins=30, rwidth=0.9,

                   color='darkturquoise')
minor=df.query("age <= 16")

print("Minors: ",minor.shape)

senior=df.query("age >=65")

print("Senior citizens: ",senior.shape)
plt.figure(figsize= (10,7))

splot= sns.countplot(x='armed', hue= 'threat_level',data=minor,palette='YlGnBu')

splot.legend(loc='upper right')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



sns.set_style('ticks')

plt.title("Minors and Armed")

plt.xlabel('Armed')

plt.ylabel('No. of deaths')

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize= (10,7))

splot= sns.countplot(x='armed',hue = 'threat_level',data=senior,palette='YlGnBu')

splot.legend(loc='upper right')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

sns.set_style('ticks')

plt.title("Seniors and Armed")

plt.xlabel('Armed')

plt.ylabel('No. of deaths')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize= (10,7))

splot= sns.countplot(x='threat_level',hue ='body_camera',data=df.query("armed=='unarmed'"),palette='YlGnBu')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

sns.set_style('ticks')

plt.title("Unarmed and threat level")

plt.xlabel('Flee')

plt.ylabel('No. of deaths')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize= (6,7))

splot= sns.countplot(data=df.query("signs_of_mental_illness == True").query("threat_level!= 'attack'"), x='flee',hue='year',palette='YlGnBu')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



sns.set_style('ticks')

splot.legend(loc='upper right')

plt.title("Mentally ill and were not attacking")

plt.xlabel('flee')

plt.ylabel('No. of deaths')

plt.xticks(rotation=45)

plt.show()
#creaet a dataframe containing states and count of killings those states.



state_count=df[['state','id']].groupby('state',as_index = False).count()

state_count.rename(columns={"id":"count"},inplace=True)
# Load the shape of the zone (US states)

# Find the original file here: https://github.com/python-visualization/folium/tree/master/examples/data

# You have to download this file and set the directory where you saved it

#url to get data of the state boundaries of USA

url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'

state_geo = f'{url}/us-states.json'

 

# Initialize the map:

m = folium.Map(location=[37, -102], zoom_start=4)

 

# Add the color for the chloropleth:

choropleth = folium.Choropleth(

 geo_data=state_geo,

 name='choropleth',

 data=state_count,

 columns=['state', 'count'],   

 key_on='feature.id',

 fill_color='GnBu',

 fill_opacity=0.7,

 line_opacity=0.2,

 legend_name='Number of shootins'

).add_to(m)

folium.LayerControl().add_to(m)



choropleth.geojson.add_child(folium.features.GeoJsonTooltip(fields = ['name'],aliases=['State'],style=('background-color: grey; color: white;')))



m
city=df[['city','id']].groupby('city',as_index = False).count()

city.rename(columns={"id":"count"},inplace=True)

city.sort_values(by='count', ascending=False,inplace=True)

city=city[:10]

city