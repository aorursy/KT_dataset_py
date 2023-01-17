import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/crime-cleanedcsv/crime.csv')

df.head()
print(df.shape, df.drop_duplicates().shape)

df = df.drop_duplicates()
df.info()
df.describe()
df["SHOOTING"].fillna("N", inplace = True)

df["DAY_OF_WEEK"] = pd.Categorical(df["DAY_OF_WEEK"], 

              categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],

              ordered=True)

df["Lat"].replace(-1, None, inplace=True)

df["Long"].replace(-1, None, inplace=True)



df.describe()
def getDate(dateStr, numChar):

    return dateStr[0:numChar]



df['DATE'] = df['OCCURRED_ON_DATE'].apply(getDate, numChar = 10)

df['YEARMONTH'] = df['OCCURRED_ON_DATE'].apply(getDate, numChar = 7)



df[['YEARMONTH', 'DATE', 'OCCURRED_ON_DATE']].head()
print('Num of records: {} \nNum of events: {}'.format(df.shape[0], df["INCIDENT_NUMBER"].nunique()))
print("It  seemes like there are {} records in database per 1 crime.".format(round(df.shape[0]/df["INCIDENT_NUMBER"].nunique(),2)))
tmp = df.groupby("INCIDENT_NUMBER")["YEAR"].count().sort_values(ascending = False)

tmp.head(10)
tmp.value_counts() #Index: num of records per crime, Values: num of occurences of such a case.
print('It occurs that {}% of our events are "duplicated" at least 2 times.'.format(round(100*(282517 - 254996) / 282517),2))
df[df["INCIDENT_NUMBER"] == "I162030584"]
df[df["INCIDENT_NUMBER"] == "I152080623"]
timeOccurencesNormal = df[['INCIDENT_NUMBER','OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'SHOOTING',

                           'DAY_OF_WEEK', 'HOUR', 'DATE', 'YEARMONTH']]

timeOccurencesDedup  = df[['INCIDENT_NUMBER','OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'SHOOTING',

                           'DAY_OF_WEEK', 'HOUR', 'DATE', 'YEARMONTH']].drop_duplicates()



print('Sanity check for duplicates: ({}, {})'.format(df['INCIDENT_NUMBER'].nunique(), timeOccurencesDedup.shape[0]))
fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (18, 10))



sns.countplot(timeOccurencesNormal["YEAR"], color='lightblue', ax = axes[0,0] )

axes[0,0].set_title("Number of crimes")

sns.countplot(timeOccurencesNormal["DAY_OF_WEEK"], color='lightgreen', ax = axes[1,0])

axes[1,0].set_title("Number of crimes")

sns.countplot(timeOccurencesNormal["HOUR"], color = 'orange', ax = axes[2,0])

axes[2,0].set_title("Number of crimes")



sns.countplot(timeOccurencesDedup["YEAR"], color='lightblue', ax = axes[0,1] )

axes[0,1].set_title("Number of crimes (deduplicated)")

sns.countplot(timeOccurencesDedup["DAY_OF_WEEK"], color='lightgreen', ax = axes[1,1] )

axes[1,1].set_title("Number of crimes (deduplicated)")

sns.countplot(timeOccurencesDedup["HOUR"], color = 'orange', ax = axes[2,1] )

axes[2,1].set_title("Number of crimes (deduplicated)")



plt.tight_layout()
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))



sns.heatmap(pd.pivot_table(data = timeOccurencesNormal, index = "DAY_OF_WEEK", 

                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count'), 

               cmap = 'Reds', ax = axes[0])

sns.heatmap(pd.pivot_table(data = timeOccurencesDedup, index = "DAY_OF_WEEK", 

                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count')

               , cmap = 'Reds', ax = axes[1])
fig = plt.figure(figsize=(18,6))



axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(timeOccurencesNormal.groupby('DATE').count(), 

          c = 'blue', label = "Original data")

axes.plot(timeOccurencesDedup.groupby('DATE').count(), 

          c = 'green', label = "Dedup data")

plt.xticks(rotation = 90)

plt.legend()

axes.set_title("Number of crimes in a day")

axes.set_ylabel("Number of crimes")



labelsX = timeOccurencesNormal.groupby('DATE').count().index[::30]

plt.xticks(labelsX, rotation='vertical')



#I've got duplicated legend here, so I used remedy:

# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib

handles, labels = axes.get_legend_handles_labels() 

i = np.arange(len(labels))

filter = np.array([])

unique_labels = list(set(labels))

for ul in unique_labels:

    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 

    

handles = [handles[int(f)] for f in filter] 

labels = [labels[int(f)] for f in filter]

axes.legend(handles, labels) 
fig = plt.figure(figsize=(18,6))



axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(timeOccurencesNormal.groupby('YEARMONTH').count(), 

          c = 'blue', label = "Original data")

axes.plot(timeOccurencesDedup.groupby('YEARMONTH').count(), 

          c = 'green', label = "Dedup data")

plt.xticks(rotation = 90)

plt.legend()

axes.set_title("Number of crimes in a month")

axes.set_ylabel("Number of crimes")



#I've got duplicated legend here, so I used remedy:

# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib

handles, labels = axes.get_legend_handles_labels() 

i = np.arange(len(labels))

filter = np.array([])

unique_labels = list(set(labels))

for ul in unique_labels:

    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 

    

handles = [handles[int(f)] for f in filter] 

labels = [labels[int(f)] for f in filter]

axes.legend(handles, labels)  
print('We have {}% of shooting crimes in all events (deduplicated situation).'.format(

    round(100*timeOccurencesDedup[timeOccurencesDedup['SHOOTING'] == 'Y'].shape[0]/timeOccurencesDedup.shape[0],2)))
fig = plt.figure(figsize=(18,6))



axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"].groupby('DATE').count(), 

          c = 'lightblue', label = "Original data")

axes.plot(timeOccurencesDedup[timeOccurencesDedup["SHOOTING"] == "Y"].groupby('DATE').count(), 

          c = 'black', label = "Dedup data")

plt.xticks(rotation = 90)

plt.legend()

axes.set_title("Shooting crimes")

axes.set_ylabel("Number of crimes with shooting")



labelsX = timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"].groupby('DATE').count().index[::30]

plt.xticks(labelsX, rotation='vertical')



#I've got duplicated legend here, so I used remedy:

# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib

handles, labels = axes.get_legend_handles_labels() 

i = np.arange(len(labels))

filter = np.array([])

unique_labels = list(set(labels))

for ul in unique_labels:

    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 

    

handles = [handles[int(f)] for f in filter] 

labels = [labels[int(f)] for f in filter]

axes.legend(handles, labels) 
fig = plt.figure(figsize=(18,6))



axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"].groupby('YEARMONTH').count(), 

          c = 'blue', label = "Original data", marker = "o")

axes.plot(timeOccurencesDedup[timeOccurencesDedup["SHOOTING"] == "Y"].groupby('YEARMONTH').count(), 

          c = 'green', label = "Dedup data", marker="o")

plt.xticks(rotation = 90)

plt.legend()

axes.set_title("Shooting crimes")

axes.set_ylabel("Number of crimes with shooting")



#I've got duplicated legend here, so I used remedy:

# https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib

handles, labels = axes.get_legend_handles_labels() 

i = np.arange(len(labels))

filter = np.array([])

unique_labels = list(set(labels))

for ul in unique_labels:

    filter = np.append(filter, [i[np.array(labels) == ul][0]]) 

    

handles = [handles[int(f)] for f in filter] 

labels = [labels[int(f)] for f in filter]

axes.legend(handles, labels) 
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))



sns.heatmap(pd.pivot_table(data = timeOccurencesNormal[timeOccurencesNormal["SHOOTING"] == "Y"], index = "DAY_OF_WEEK", 

                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count'), 

               cmap = 'Reds', ax = axes[0])

sns.heatmap(pd.pivot_table(data = timeOccurencesDedup[timeOccurencesNormal["SHOOTING"] == "Y"], index = "DAY_OF_WEEK", 

                              columns = "HOUR", values = "INCIDENT_NUMBER", aggfunc = 'count')

               , cmap = 'Reds', ax = axes[1])
locationOccurencesNormal = df[['INCIDENT_NUMBER','DISTRICT', 'REPORTING_AREA', 'SHOOTING','Lat', 'Long']]

locationOccurencesDedup  = df[['INCIDENT_NUMBER','DISTRICT', 'REPORTING_AREA', 'SHOOTING','Lat', 'Long']].drop_duplicates()

print('Sanity check for duplicates: ({}, {})'.format(df['INCIDENT_NUMBER'].nunique(), locationOccurencesDedup.shape[0]))
# Plot districts

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))

sns.scatterplot(y='Lat',

                x='Long',

                hue='DISTRICT',

                alpha=0.01,

                data=locationOccurencesNormal, 

                ax = axes[0])

#plt.ylim(locationOccurencesNormal['Long'].max(), locationOccurencesNormal['Long'].min())

sns.scatterplot(y='Lat',

                x='Long',

                hue='DISTRICT',

                alpha=0.01,

                data=locationOccurencesDedup, 

                ax = axes[1])
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 7))

sns.scatterplot(y = 'Lat',

                x = 'Long',

                alpha = 0.3,

                data = locationOccurencesNormal[locationOccurencesNormal["SHOOTING"]=="Y"], 

                ax = axes[0])

axes[0].set_title("Crime locations")

sns.scatterplot(y = 'Lat',

                x = 'Long',

                alpha = 0.3,

                data = locationOccurencesDedup[locationOccurencesDedup["SHOOTING"]=="Y"], 

                ax = axes[1])

axes[1].set_title("Crime locations (deduplicated)")
# Below I choose a YEAR column cause I would like to narrow the data processed 

# and this columns is nice -> doesn't have any null values 

tmp = df.groupby('INCIDENT_NUMBER')['YEAR'].count().sort_values(ascending = False)

tmp = pd.DataFrame({'INCIDENT_NUMBER': tmp.index, 'NUM_RECORDS': tmp.values})

seriousCrimes = df.merge(tmp[tmp['NUM_RECORDS'] > 5], on = 'INCIDENT_NUMBER', how = 'inner')

seriousCrimes = seriousCrimes[['INCIDENT_NUMBER', 'Lat','Long']].drop_duplicates()[['Lat','Long']].dropna()
#!pip install folium

# Used this tutorial: https://medium.com/@bobhaffner/folium-markerclusters-and-fastmarkerclusters-1e03b01cb7b1

import folium

from folium.plugins import MarkerCluster



some_map = folium.Map(location = [seriousCrimes['Lat'].mean(), 

                                  seriousCrimes['Long'].mean()], 

                      zoom_start = 12)

mc = MarkerCluster()

#creating a Marker for each point. 

for row in seriousCrimes.itertuples():

    mc.add_child(folium.Marker(location = [row.Lat,  row.Long]))



some_map.add_child(mc)



some_map
locationTimeOccurencesDedup = df[['INCIDENT_NUMBER', 'SHOOTING','Lat', 'Long', 'DAY_OF_WEEK', 'HOUR']].drop_duplicates()

print('Sanity check for duplicates: ({}, {})'.format(df['INCIDENT_NUMBER'].nunique(), locationOccurencesDedup.shape[0]))
fig = plt.figure(figsize=(18,10))

g = sns.FacetGrid(data = locationTimeOccurencesDedup[(locationTimeOccurencesDedup['HOUR'] >= 1) & (locationTimeOccurencesDedup['HOUR'] <= 7)],

                                                  row = 'HOUR', col = 'DAY_OF_WEEK')

g = g.map(sns.scatterplot, 'Long', 'Lat', alpha=0.03)