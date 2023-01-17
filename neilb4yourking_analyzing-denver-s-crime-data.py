#Start by importing some useful packages



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy import stats

import numpy as np

import folium

from folium import plugins

#Next, read in the data, publicly available at https://www.denvergov.org/opendata/dataset/city-and-county-of-denver-crime

denver_data=pd.read_csv('../input/crime.csv', parse_dates=True)

denver_data.head(8)
#Check to see how many non-null values present

denver_data.info()
temp=display(denver_data.groupby([denver_data.OFFENSE_CODE,denver_data.OFFENSE_CODE_EXTENSION,denver_data.OFFENSE_TYPE_ID]).size())

pd.set_option('display.max_rows',500)

print(temp)
#convert relevant column data into datetime objects

denver_data['REPORTED_DATE']=pd.to_datetime(denver_data.REPORTED_DATE)

denver_data['FIRST_OCCURRENCE_DATE']=pd.to_datetime(denver_data.FIRST_OCCURRENCE_DATE)

denver_data['LAST_OCCURRENCE_DATE']=pd.to_datetime(denver_data.LAST_OCCURRENCE_DATE)

# Calculate the time difference between first and last occurence date. negative time means the Last Occurence is dated

# before first occurence. those entries have been listed with OFFENSE_ID for easy access

temp=denver_data[['OFFENSE_ID','FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','REPORTED_DATE']]

temp.loc[:,'OCCURENCE_WINDOW']=temp.LAST_OCCURRENCE_DATE-temp.FIRST_OCCURRENCE_DATE

temp.loc[:,'OCCURENCE_WINDOW']=temp.OCCURENCE_WINDOW.fillna(0)

print(temp[temp['OCCURENCE_WINDOW']<'0'])

duds=temp[temp['OCCURENCE_WINDOW']<'0'].OFFENSE_ID

#remove rows with errors

denver_data=denver_data[~denver_data['OFFENSE_ID'].isin(duds)]

#print the OFFENSE_ID's of rows wih errors

#check that reported dates are after first/last occurence dates

temp=denver_data[['OFFENSE_ID','FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','REPORTED_DATE']]

temp.loc[:,'OCC_REPORT_GAP']=temp.REPORTED_DATE-temp.LAST_OCCURRENCE_DATE

temp.loc[:,'OCC_REPORT_GAP']=temp.OCC_REPORT_GAP.fillna(temp.REPORTED_DATE-temp.FIRST_OCCURRENCE_DATE)

#list all instances where reported date is before either of the occurence dates

print(temp[temp['OCC_REPORT_GAP']<'0'])

duds2=temp[temp['OCC_REPORT_GAP']<'0'].OFFENSE_ID

denver_data=denver_data[~denver_data['OFFENSE_ID'].isin(duds2)]

#Determine if there are mistakes in the OFFENSE_ID column

#prepare the columns

temp=denver_data[['OFFENSE_ID','INCIDENT_ID','OFFENSE_CODE','OFFENSE_CODE_EXTENSION']]

temp.OFFENSE_ID=temp.OFFENSE_ID.astype(str)

temp.INCIDENT_ID=temp.INCIDENT_ID.astype(str)

#the below two columns have to be homogonized to 4 digits and 2 digits respectively

temp.OFFENSE_CODE=temp.OFFENSE_CODE.map('{:04d}'.format).astype(str)

temp.OFFENSE_CODE_EXTENSION=temp.OFFENSE_CODE_EXTENSION.map('{:02d}'.format).astype(str)

#Combine the composite columns and check to make sure it's the same as the OFFENSE_ID

temp['COMBINED_ID']=temp.INCIDENT_ID+temp.OFFENSE_CODE+temp.OFFENSE_CODE_EXTENSION

temp['ID_MATCH']=temp.COMBINED_ID==temp.OFFENSE_ID

#print all rows which have mistakes. will return no rows if there are no errors

print(temp[temp['ID_MATCH']==False])



#filter out traffic accidents from the crime dataset

denver_crime=denver_data[denver_data['IS_CRIME']==1]

denver_crime=denver_crime[denver_crime['REPORTED_DATE']<'2019']

#add columns indicating the hour, day, month, and year eeach crime occurred

denver_crime['HOUR_REPORTED']=pd.DatetimeIndex(denver_crime['REPORTED_DATE']).hour

denver_crime['WEEKDAY_REPORTED']=pd.DatetimeIndex(denver_crime['REPORTED_DATE']).weekday

denver_crime['MONTH_REPORTED']=pd.DatetimeIndex(denver_crime['REPORTED_DATE']).month

denver_crime['YEAR_REPORTED']=pd.DatetimeIndex(denver_crime['REPORTED_DATE']).year



denver_crime

#How has the amount of crime in Denver changed these past five years?

denver_crime['YEAR_REPORTED'].groupby(denver_crime.YEAR_REPORTED).agg('count').plot('line')

plt.xlabel('Year')

plt.ylabel('Number of Crimes')

plt.title('Crimes Trends by Year')

plt.ylim(bottom=0,top=80000)

plt.show()
denver_crime['OFFENSE_CATEGORY_ID'].value_counts().plot(kind='bar')

plt.title('Crimes Committed by Type')

plt.xlabel('Offense Type')

plt.ylabel('Total Crimes (2014-2018)')

plt.show()
#only listing the top thirteen subdivisions of 'all-other-crimes', for sake of space

temp=denver_crime[denver_crime['OFFENSE_CATEGORY_ID']=='all-other-crimes'].OFFENSE_TYPE_ID.value_counts().head(13).plot(kind='bar')

plt.title('All Other Crimes Expanded')

plt.xlabel('Crime Type')

plt.ylabel('Total Crimes (2014-2018)')

plt.show()
denver_misc_crimes=denver_crime[denver_crime['OFFENSE_CATEGORY_ID']=='other-crimes-against-persons'].OFFENSE_TYPE_ID.value_counts().plot(kind='bar')

plt.title('Crimes Against Persons Expanded')

plt.xlabel('Crime Type')

plt.ylabel('Total Crimes (2014-2018)')

plt.show()
temp=denver_crime[['YEAR_REPORTED','OFFENSE_CATEGORY_ID']].groupby([denver_crime.YEAR_REPORTED, denver_crime.OFFENSE_CATEGORY_ID]).agg('count')

temp=temp.drop(labels='YEAR_REPORTED',axis=1)

temp.columns=['COUNT']

temp=temp.unstack(level=1)

temp.plot(kind='line')

plt.legend(loc=(1.2,0))

plt.ylabel('TOTAL')

plt.title('Crime Types over the Years')

plt.show()
# br graph of crimes by month, to see if crime is seasonal

temp=denver_crime[['MONTH_REPORTED','OFFENSE_CATEGORY_ID']].groupby([denver_crime.MONTH_REPORTED, denver_crime.OFFENSE_CATEGORY_ID]).agg('count')

temp=temp.drop(labels='MONTH_REPORTED',axis=1)

temp.columns=['COUNT']

temp=temp.unstack(level=1)

ax=temp.plot(kind='bar',stacked=True)

plt.legend(loc=(1.2,0))

plt.title('Total Crimes each Month (Bar Graph)')

plt.xlabel('Month')

plt.ylabel('TOTAL')

ax.set_xticklabels(('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

plt.show()
#line graph of crimes by month to see if certain crimes increase noticeably in certain months

temp=denver_crime[['MONTH_REPORTED','OFFENSE_CATEGORY_ID']].groupby([denver_crime.MONTH_REPORTED, denver_crime.OFFENSE_CATEGORY_ID]).agg('count')

temp=temp.drop(labels='MONTH_REPORTED',axis=1)

temp.columns=['COUNT']

temp=temp.unstack(level=1)

ax=temp.plot(kind='line')

plt.legend(loc=(1.2,0))

plt.title('Total Crimes each Month(Line Graph)')

plt.xlabel('Month')

plt.ylabel('TOTAL')

ax.set_xticklabels(('','Feb','Apr','Jun','Aug','Oct','Dec'))



plt.show()
summer=denver_crime[(denver_crime['MONTH_REPORTED']==8)|(denver_crime['MONTH_REPORTED']==6)|(denver_crime['MONTH_REPORTED']==7)]

winter=denver_crime[(denver_crime['MONTH_REPORTED']==12)|(denver_crime['MONTH_REPORTED']==1)|(denver_crime['MONTH_REPORTED']==2)]

summer=summer.REPORTED_DATE.dt.date.groupby(summer.REPORTED_DATE.dt.date).agg('count')

winter=winter.REPORTED_DATE.dt.date.groupby(winter.REPORTED_DATE.dt.date).agg('count')

stats.ttest_ind(summer,winter,equal_var=False)

#bar graph of total crimes by day of the week

temp=denver_crime[['WEEKDAY_REPORTED','OFFENSE_CATEGORY_ID']].groupby([denver_crime.WEEKDAY_REPORTED, denver_crime.OFFENSE_CATEGORY_ID]).agg('count')

temp=temp.drop(labels='WEEKDAY_REPORTED',axis=1)

temp.columns=['COUNT']

temp=temp.unstack(level=1)

ax=temp.plot(kind='bar',stacked=True)

plt.legend(loc=(1.2,0))

plt.xlabel('Day of the Week(bar graph)')

plt.ylabel('TOTAL')

plt.title('Crimes by Day of the Week')

ax.set_xticklabels(('Mon','Tue','Wed','Thu','Fri','Sat','Sun'))

plt.show()







#line graph of each type of crime by day of the week

temp=denver_crime[['WEEKDAY_REPORTED','OFFENSE_CATEGORY_ID']].groupby([denver_crime.WEEKDAY_REPORTED, denver_crime.OFFENSE_CATEGORY_ID]).agg('count')

temp=temp.drop(labels='WEEKDAY_REPORTED',axis=1)

temp.columns=['COUNT']

temp=temp.unstack(level=1)

ax=temp.plot(kind='line')

plt.legend(loc=(1.2,0))

plt.title('Crimes by Day of the Week (line graph)')

plt.xlabel('Day of the Week')

plt.ylabel('TOTAL')

ax.set_xticklabels(('','Mon','Tue','Wed','Thu','Fri','Sat','Sun'))

plt.show()
weekday=denver_crime[(denver_crime['WEEKDAY_REPORTED']<5)]

weekend=denver_crime[(denver_crime['WEEKDAY_REPORTED']>=5)]

weekday=weekday.REPORTED_DATE.dt.date.groupby(weekday.REPORTED_DATE.dt.date).agg('count')

weekend=weekend.REPORTED_DATE.dt.date.groupby(weekend.REPORTED_DATE.dt.date).agg('count')

stats.ttest_ind(weekday,weekend,equal_var=False)
#create lineplot to see crime trends of total crimes throughout the day

temp=denver_crime[['HOUR_REPORTED']].groupby([denver_crime.HOUR_REPORTED]).agg('count')

temp.plot(kind='line')

plt.xlabel('Hour')

plt.ylabel('TOTAL')

plt.title('Crimes by Hour of Day')

plt.show()


temp=denver_crime[['HOUR_REPORTED','OFFENSE_CATEGORY_ID']].groupby([denver_crime.HOUR_REPORTED, denver_crime.OFFENSE_CATEGORY_ID]).agg('count')

temp=temp.drop(labels='HOUR_REPORTED',axis=1)

temp.columns=['COUNT']

temp.index=temp.index.rename('HOUR',level=0)



temp=temp.unstack(level=1)

fig=temp.plot(kind='line', figsize=(18,30),subplots=True,layout=(-1,3),sharex=False,sharey=False)

for row in fig:

    for item in row:

        item.set_ylabel('TOTAL')

        item.legend(loc='upper left')

plt.suptitle('Crimes by Hour of the Day')

plt.show()



#analyze crime by district in each of the past five years

temp=denver_crime[['REPORTED_DATE','DISTRICT_ID']].groupby([denver_crime.REPORTED_DATE.dt.year, denver_crime.DISTRICT_ID]).agg('count')

temp=temp.drop(labels='REPORTED_DATE',axis=1)

temp.columns=['TOTAL']

temp=temp.unstack(level=0)

temp.plot(kind='bar',stacked=False)

plt.ylabel('TOTAL')

plt.title('Crimes by District')

plt.legend(loc=(1.2,0))

plt.show()


    
temp=denver_crime[['REPORTED_DATE','DISTRICT_ID','OFFENSE_CATEGORY_ID']]

for offense in denver_crime.OFFENSE_CATEGORY_ID.unique():

    temp2=temp[temp['OFFENSE_CATEGORY_ID']==offense].drop('OFFENSE_CATEGORY_ID', axis=1)

    temp2=temp2=temp2.groupby([temp2.REPORTED_DATE.dt.year,temp2.DISTRICT_ID]).agg('count')

    temp2.columns= ['TOTAL']

    temp2=temp2.unstack(level=0)

    temp2=temp2.plot(kind='bar')

    plt.ylabel('TOTAL')

    plt.legend(loc=(1.2,0))

    plt.title(offense)

    

# This function lets you create your own heatmaps using the provided latitude and longitude coordinates

# the function parameters filter denver_crime by its columns. pass in a tuple of (column_name,column_value) and it 

# will create a heatmap with the entries that satisfy the condition. You can have as many filters as you'd like

def heatmap_creator(*args):

    #filter out all entries with no Lat/Lon data

    check_for_coordinates=denver_crime['GEO_LAT'].isna()|denver_crime['GEO_LON'].isna()

    temp=denver_crime[~check_for_coordinates]

    #filter based on the provided parameters

    for item in args:

        a,b=item

        temp=temp[temp[a]==b]

        print(str(a)+': '+str(b))

    #Generate heatmap

    hm_prep=temp[['GEO_LAT','GEO_LON']].as_matrix()

    m = folium.Map(location=[39.73,-104.90], tiles='Stamen Toner',zoom_start=11, control_scale=True)

    m.add_children(plugins.HeatMap(hm_prep,radius=15))

    return display(m)

heatmap_creator(('OFFENSE_CATEGORY_ID','drug-alcohol'))
heatmap_creator(('MONTH_REPORTED',7),('YEAR_REPORTED',2015),('OFFENSE_CATEGORY_ID','drug-alcohol'))
#all-other-crimes over the years

temp=denver_crime[denver_crime['OFFENSE_CATEGORY_ID']=='all-other-crimes']

temp=temp[['YEAR_REPORTED','OFFENSE_TYPE_ID']].groupby([temp.OFFENSE_TYPE_ID, temp.YEAR_REPORTED]).agg('count')

temp=temp.drop(labels='YEAR_REPORTED',axis=1)

temp.columns=['COUNT']

temp=temp.unstack(level=0)

temp['COUNT'][['criminal-trespassing','traf-other','police-false-information','public-order-crimes-other']].plot(kind='line')

plt.legend(loc=(1.2,0))

plt.ylabel('TOTAL')

plt.title('all-other-crimes over the Years')

plt.show()



#crime breakdown within each district

temp=denver_crime[['REPORTED_DATE','DISTRICT_ID','OFFENSE_CATEGORY_ID']]

for district in range(1,8):

    temp2=temp[temp['DISTRICT_ID']==district].drop('DISTRICT_ID',axis=1)

    temp2=temp2.groupby([temp2.REPORTED_DATE.dt.year,temp2.OFFENSE_CATEGORY_ID]).agg('count')

    temp2=temp2.unstack(level=0).plot(kind='bar').legend(loc=(1.2,0))

    plt.title('District '+str(district))

    plt.show()
denver_traffic=denver_data[denver_data['IS_TRAFFIC']==1]

check_for_coordinates=denver_traffic['GEO_LAT'].isna()|denver_traffic['GEO_LON'].isna()

temp=denver_traffic[~check_for_coordinates]

hm_prep=temp[['GEO_LAT','GEO_LON']].as_matrix()

m = folium.Map(location=[39.73,-104.90], tiles='Stamen Toner',zoom_start=11, control_scale=True)

m.add_children(plugins.HeatMap(hm_prep,radius=15))

display(m)
denver_crime[['DISTRICT_ID','NEIGHBORHOOD_ID']].groupby(['DISTRICT_ID','NEIGHBORHOOD_ID']).agg('count')