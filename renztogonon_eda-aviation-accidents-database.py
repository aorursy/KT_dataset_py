import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cartopy.crs as ccrs

import cartopy.feature as cf

#load data

data = pd.read_csv('../input/aviation-accident-database-synopses/AviationData.csv',  engine = 'python')

data.head(10)
# no impute, visualize available data



#SOME IDEAS:

# event ID dupes remove

# accident number remove

# feature engineer event date

# feature engineer LOC (state?) (show map?)

# Lat Long

# injury severity pretty straightforward          

# format 'Make', some CESSNA, some Cessna*

# model related to format, check most frequent values

# drop sched 70k unknown

# drop air carrier 80k unkown

# search report status meaning
#feature engineer 'Make'

#data['Make'].str.lower().capitalize()

data['Make'] = data['Make'].apply(lambda x: x.lower().capitalize() if type(x) == str else x)
#feature engineer 'Injury.Severity', all fatal accidents into one feature regardless of count,  convert unavailable values to NaN

data['Injury.Severity']= data['Injury.Severity'].apply(lambda x: x[:5] if  '(' in x else x)

data['Injury.Severity'].loc[data['Injury.Severity'] == 'Unavailable'] = np.nan
#internal relationship visualization ideas:



# -date (monthly/day  of week/yearly histogram) lineplot

# -location (definitely a heat map on a literal map of sorts/ country) lineplot

# -heatmap worldmap

# -heatmap available lat longs

# -histogram injury severity

# -histo aircraft damage

# -histo aircraft cat  available

# -histo make get top 10 

# -histo amateur built

# -histo number  of engines

# -histo engine type

# -histo  far desc

# -histo purpose

# -histo weather condition

# -histo broad phase of flight





#list of histograms

histo_features = ['Injury.Severity','Aircraft.Damage','Aircraft.Category', 'Make','Amateur.Built','Number.of.Engines','Engine.Type','FAR.Description','Purpose.of.Flight','Weather.Condition','Broad.Phase.of.Flight']

histo_obj =[x  for x in histo_features if data[x].dtypes == 'object']

histo_num =[x  for x in histo_features if data[x].dtypes != 'object']
#histograms

fig  = plt.figure(figsize = (12,34))

for x in  histo_features:

    fig.add_subplot(6,2, histo_features.index(x)+1)

    if len(data[x].value_counts()) >10:

        sns.countplot(x = data[x].dropna(), 

                      data  =  data, 

                      order = data[x].value_counts()[0:9].index).set(xlabel= x, 

                                                                     ylabel = "Accident Count 1940 - 2020")

        plt.xticks(rotation=90)

    else:

        sns.countplot(x = data[x].dropna(), 

                      data  =  data).set(xlabel= x, 

                                         ylabel = "Accident Count 1940 - 2020")

        plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
#create df of existing coordinates

coordinatesdf =  data[['Latitude', 'Longitude','Injury.Severity']]

coordinatesdf = coordinatesdf.dropna(axis = 0, 

                                     subset = ['Longitude','Latitude'])
#lat long projection of aircraft  crash

'''fig = plt.figure(figsize =  (30,19))

ax = fig.add_subplot(1,1,1, 

                     projection = ccrs.PlateCarree())

ax.add_feature(cf.LAND)

ax.add_feature(cf.OCEAN)

ax.add_feature(cf.COASTLINE)

ax.add_feature(cf.BORDERS, linestyle=':')

ax.add_feature(cf.LAKES, alpha=0.5)

ax.add_feature(cf.RIVERS)

ax.set_title("Aircraft Accidents", fontsize =  50)

sns.scatterplot(x = 'Longitude', y  = 'Latitude', data= coordinatesdf,

                hue =  'Injury.Severity',

                alpha  =  0.5,

                marker  = 'o',

                s = 100,

                color ='red',

                transform=ccrs.PlateCarree())

plt.show()

'''
#feature engineer Dates  for visualization

data['parsedate'] = pd.to_datetime(data['Event.Date'])

data['Day.Of.Week'] = data['parsedate'].dt.day_name()

data['Month.Name'] = data['parsedate'].dt.month_name()

data['year'] = data['Event.Date'].str[0:4].astype(int)
#year histogram

sns.distplot(a =  data['year'], 

             bins  = 72,

             kde= True)

plt.xticks(rotation = 90)
fig = plt.figure(figsize = (18,7))









fig.add_subplot(1,2,1)

sns.countplot(x = 'Month.Name', 

              data = data, 

              order  =  ['January', 'February','March', 'April','May',  

                         'June','July', 'August', 'September', 'October','November','December']).set(xlabel= 'Month Name', 

                                                                                                     ylabel = "Accident Count 1940 - 2020")

plt.xticks(rotation = 90)



fig.add_subplot(1,2,2)

sns.countplot(x = 'Month.Name', 

              data = data,  

              order = data['Month.Name'].value_counts()[:].index).set(xlabel= 'Month Name (Greatest  to Least)', 

                                                                      ylabel = "Accident Count 1940 - 2020")

plt.xticks(rotation = 90)
sns.countplot(x = 'Day.Of.Week', data = data, 

              order  =  ['Sunday', 'Monday','Tuesday', 'Wednesday','Thursday','Friday','Saturday']).set(xlabel = 'Day of Week',

                                                                                                        ylabel = 'Accident Count 1940 - 2020')

plt.xticks(rotation = 90)
#relationship between features:



# -amateur built/broad phase of flight

# -aircraft damage/phase of flight (mirror bar graph?) modify yticks maybe*

# -weather condition/phase of flight (mirror bar graph?)*

# -(make/4 types of engine) count of accidents

# - injury severity(AVERAGE fatalities)/engine type bar graph
# amateur built/broad phase of flight

fig = plt.figure(figsize = (18,7))

fig.add_subplot(1,2,1)

data[['Amateur.Built','Broad.Phase.of.Flight']]

sns.countplot(x = 'Broad.Phase.of.Flight', 

              hue  = 'Amateur.Built', 

              data =data[~data['Broad.Phase.of.Flight'].isin(['UNKNOWN', 'OTHER'])].dropna(subset = ['Broad.Phase.of.Flight'])).set(xlabel= 'Broad Phase of Flight', 

                                                                                                                                    ylabel = "Accident Count 1940 - 2020")

plt.legend(title='Amateur Built', loc='upper right', labels=['No', 'Yes'])

plt.xticks(rotation  =  90)

fig.add_subplot(1,2,2)

sns.countplot(x = data['Amateur.Built'].dropna(), 

                      data  =  data, 

                      order = data['Amateur.Built'].value_counts()[0:9].index).set(xlabel= 'Amateur Built', 

                                                                                   ylabel = "Accident Count 1940 - 2020")
# aircraft damage/phase of flight

fig = plt.figure(figsize = (18,7))

fig.add_subplot(1,2,1)

sns.countplot(x = 'Broad.Phase.of.Flight', 

              hue  = 'Aircraft.Damage', 

              data =data[~data['Broad.Phase.of.Flight'].isin(['UNKNOWN', 'OTHER'])].dropna(subset = ['Broad.Phase.of.Flight'])).set(xlabel= 'Broad Phase of Flight', 

                                                                                                                                    ylabel = "Accident Count 1940 - 2020")

plt.xticks(rotation  =  90)

plt.legend(title='Aircraft Damage', loc='upper right', labels=['Substantial', 'Destroyed','Minor'])

fig.add_subplot(1,2,2)

sns.countplot(x = data['Aircraft.Damage'].dropna(), 

                      data  =  data, 

                      order = data['Aircraft.Damage'].value_counts()[0:9].index).set(xlabel= 'Aircraft Damage', 

                                                                                   ylabel = "Accident Count 1940 - 2020")



# weather condition/phase of flight (mirror bar graph?)

fig = plt.figure(figsize = (18,7))

fig.add_subplot(1,2,1)

sns.countplot(x = 'Broad.Phase.of.Flight', 

              hue  = 'Weather.Condition', 

              data =data[~data['Broad.Phase.of.Flight'].isin(['UNKNOWN', 'OTHER'])].dropna(subset = ['Broad.Phase.of.Flight'])).set(xlabel= 'Broad Phase of Flight', 

                                                                                                                                    ylabel = "Accident Count 1940 - 2020") 

plt.xticks(rotation  =  90)

plt.legend(title='Weather Condition')

fig.add_subplot(1,2,2)

sns.countplot(x = data['Weather.Condition'].dropna(), 

                      data  =  data, 

                      order = data['Weather.Condition'].value_counts()[0:9].index).set(xlabel= 'Weather Condition', 

                                                                                       ylabel = "Accident Count 1940 - 2020")
sns.violinplot(x = 'Weather.Condition', 

               y = 'year',

               hue = 'Injury.Severity',

               split  =  True,

               data = data[data['Injury.Severity'] != 'Incident'].dropna(subset  = ['Weather.Condition','year','Injury.Severity'], axis = 0))
# (make/4 types of engine) count of accidents

fig = plt.figure(figsize = (18,7))



fig.add_subplot(1,2,1)

sns.countplot(x = 'Make', 

              data  =  data.dropna(subset = ['Number.of.Engines']), 

              hue = 'Number.of.Engines',

              order = data['Make'].value_counts()[0:9].index)

plt.xticks(rotation  =  90)

plt.xlabel('Make')

plt.ylabel("Accident Count 1940 - 2020")

plt.legend(title='Number of Engines',   loc = 'upper right')



fig.add_subplot(1,2,2)

sns.distplot(a = data['Number.of.Engines'].dropna(), kde=False).set(ylabel = "Accident Count 1940 - 2020")
# number of engines/ engine  type

violindf = data[data['Engine.Type'].isin(['Reciprocating','Turbo Prop', 'Turbo Shaft', 'Turbo Fan','Turbo Jet'])]



fig = plt.figure(figsize = (20,20))



fig.add_subplot(2,1,1)

sns.countplot(x = 'Engine.Type', 

              data  =  data.dropna(subset = ['Number.of.Engines']), 

              hue = 'Number.of.Engines')

plt.xticks(rotation  =  90)

plt.xlabel('Engine Type')

plt.ylabel('Accident Count 1940 - 2020')

plt.legend(title='Number of Engines',   loc = 'upper right')



fig.add_subplot(2,1,2)

sns.countplot(x = 'Engine.Type',

              hue = data['Aircraft.Category'].apply(lambda x: x if x in ['Airplane', 'Helicopter', 'Glider'] else np.nan), #for simplicity

              data  =  data.dropna(subset = ['Aircraft.Category']))

plt.xticks(rotation  =  90)

plt.xlabel('Engine Type')

plt.ylabel('Accident Count 1940 - 2020')

plt.legend(title='Aircraft Category',   loc = 'upper right')
# injury severity(Total.Fatal.Injuries)/engine type violin plot

# For engine type, I will be using the 5 main types of engine as it is  more  convenient, and it makes up 77 921 (97.4%) of non-null data.



violindf = data[data['Engine.Type'].isin(['Reciprocating','Turbo Prop', 'Turbo Shaft', 'Turbo Fan','Turbo Jet'])]

fig = plt.figure(figsize = (20,12))

fig.add_subplot(1,2,2)

sns.violinplot(x = 'Engine.Type', 

               y = 'Total.Fatal.Injuries',

               data = violindf)

plt.grid(b =True,  axis  = 'y',   linewidth = 1)

fig.add_subplot(1,2,1)

sns.regplot(x = 'Number.of.Engines', 

               y = 'Total.Fatal.Injuries',

               data = violindf)