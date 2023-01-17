# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading in csv file and changing dtype of 'Reported Date column to datetime'

mm_data = pd.read_csv('/kaggle/input/missing-migrants-project/MissingMigrants-Global-2019-12-31_correct.csv', parse_dates = ['Reported Date'])
#dataset information

mm_data.info()
#time period covered in data set

print('The Missing Migrants dataset covers the period {0} to {1}'.format(str(mm_data['Reported Date'][5986]), str(mm_data['Reported Date'][0])))
#missing values

mm_data.isnull().sum().sort_values(ascending=False)
#extracting only the rows where 'Number Dead' is null

null_number_dead = mm_data[mm_data['Number Dead'].isnull()]

print('There are {} missing values in the "Number Dead" column.'.format(null_number_dead.shape[0]))
null_number_dead.head()
#boolean mask to filter entries where number missing is not equal to total number dead and missing

bool_null_number_dead = null_number_dead[null_number_dead['Minimum Estimated Number of Missing'] != null_number_dead['Total Dead and Missing']]
bool_null_number_dead
#extracting the rows where 'Number Dead', 'Minimum Estimated Number of Missing', and 'Number of Survivors' have missing data (ie NaN).

missing_data = mm_data[mm_data['Number Dead'].isnull() & mm_data['Minimum Estimated Number of Missing'].isnull() & mm_data['Number of Survivors'].isnull()]
missing_data
#row 4226

mm_data.loc[4226, 'Number Dead'] = 3

mm_data.loc[4226, 'Total Dead and Missing'] = 3

#row 5253

mm_data.loc[5253, 'Number Dead' ] = 11

mm_data.loc[5253, 'Number of Survivors'] = 15

mm_data.loc[5253, 'Total Dead and Missing'] = 11

#row 5337

mm_data.loc[5667, 'Number Dead'] = 6

mm_data.loc[5667, 'Total Dead and Missing'] = 6

#replacing NaN with 0 in 'Number Dead' column

mm_data['Number Dead'].fillna(0, inplace=True)
#extracting only those entries where 'Minimum Estimated Number of Missing' is NaN

null_missing = mm_data[mm_data['Minimum Estimated Number of Missing'].isnull()]

print('There are {} missing values in the "Minimum Estimated Number of Missing" column.'.format(null_missing.shape[0]))
null_missing.head()
#boolean mask to filter any values where number dead is not equal to total dead and missing

bool_null_missing = null_missing[null_missing['Number Dead'] != null_missing['Total Dead and Missing']]

bool_null_missing.shape[0]
#replacing NaN with 0 in 'Minimum Estimated Number of Missing' in main dataset, mm_data

mm_data['Minimum Estimated Number of Missing'].fillna(0, inplace=True)
#extracting relevant row

null_loc_coord = mm_data[mm_data['Location Coordinates'].isnull()]

null_loc_coord
#replacing NaN with approximate location coordinate for row 3097

mm_data.loc[3097, 'Location Coordinates'] = '20.191944, 12.9675'
#any missing values left to handle for columns we are interested in?

mm_data[['Region of Incident', 'Reported Date', 'Reported Year', 

        'Reported Month', 'Number Dead', 'Minimum Estimated Number of Missing',

        'Total Dead and Missing', 'Cause of Death', 'Location Coordinates']].isnull().sum()
import matplotlib.pyplot as plt

import folium

import seaborn as sns

%matplotlib inline
#create new column of marker labels for folium map

marker_loc = mm_data['Location Description'] 

marker_date = mm_data['Reported Date'].dt.date.astype(str)

marker_number = mm_data['Total Dead and Missing'].astype(str)

marker_cause = mm_data['Cause of Death']

#adding object series into 'Marker Labels'

marker_labels = 'Location: ' + marker_loc + '; Date: '+ marker_date + '; Total Dead and Missing: ' + marker_number + '; Cause of Death: ' + marker_cause

mm_data['Marker Label'] = marker_labels
#map of incidents



from ast import literal_eval

from folium.plugins import MarkerCluster



incidents_map = folium.Map(location=[50,0], tiles = 'CartoDB dark_matter', zoom_start=3, min_zoom = 2.5, control_scale = True)



marker_cluster = MarkerCluster().add_to(incidents_map)



for i in range(mm_data.shape[0]):

    loc = list(literal_eval(mm_data.iloc[i]['Location Coordinates']))

    folium.Marker(

        location = loc,

        popup = mm_data.iloc[i]['Marker Label'], 

        tooltip = mm_data.iloc[i]['Marker Label'],

        icon=folium.Icon(color='red'),

    ).add_to(marker_cluster)





display(incidents_map)

from folium.plugins import HeatMap



#create list of lists of coordinates and Total Dead and Missing

victims_array = []

for i in range(mm_data.shape[0]):

    victims_array.append(list(literal_eval(mm_data.iloc[i]['Location Coordinates'])))

    victims_array[i].append(float(mm_data.iloc[i]['Total Dead and Missing']))





victims_map = folium.Map(location=[50,0], tiles = 'CartoDB dark_matter', zoom_start=3, min_zoom = 2.5, control_scale = True)

HeatMap(victims_array, min_opacity = 0.25).add_to(victims_map)

display(victims_map)

#number of incidents

incidents_reg_count = mm_data['Region of Incident'].value_counts()
#bar graph

sns.set(style="white")

plt.figure(figsize=(10,10))

sns.barplot(incidents_reg_count.index, incidents_reg_count.values, palette='YlOrRd_r')

plt.xlabel('Region', fontsize = 13)

plt.xticks(rotation = 90)

plt.ylabel('Number of Incidents', fontsize = 13)

plt.title('Number of Incidents of Each Region (from January 2014 to December 2019)', fontsize = 15)

plt.show()
#sort main data set in ascending order with regards to time and store in time_ordered_mmdata

time_ordered_mmdata = mm_data.sort_values(by='Reported Date', ascending=True)
#extracting portion of dataset for regions US-Mexico Border, North Africa, and the Mediterranean

top_regions_data = time_ordered_mmdata[(time_ordered_mmdata['Region of Incident'] == 'US-Mexico Border') | 

                                       (time_ordered_mmdata['Region of Incident'] == 'North Africa') | 

                                       (time_ordered_mmdata['Region of Incident'] == 'Mediterranean')]
#add column of ones called 'Number of Incidents' to top_regions_data to be able to count

top_regions_data.loc[:,'Number of Incidents'] = 1
#grouping by year and region and counting number of incidents

region_year_group = top_regions_data.pivot_table(index=['Region of Incident','Reported Year'], values='Number of Incidents', aggfunc='count')

region_year_group
#plotting as line graphs

sns.set(style = 'ticks' )



med = region_year_group.loc['Mediterranean']

n_a = region_year_group.loc['North Africa']

us_m =region_year_group.loc['US-Mexico Border']



plt.figure(figsize=(10,10))



plt.plot(med['Number of Incidents'], 'r:' ,label ='Mediterranean', marker = 'o', markersize = 5, mew = 2, linewidth = 3)

plt.plot(n_a['Number of Incidents'], 'g:' ,label ='North Africa', marker = 'o', markersize = 5, mew = 2, linewidth = 3)

plt.plot(us_m['Number of Incidents'], 'b:', label ='US-Mexico Border', marker = 'o', markersize = 5, mew = 2, linewidth = 3)



plt.xlabel('Year', fontsize = 13)

plt.ylabel('Number of Incidents', fontsize = 13)

plt.title('Number of Incidents by Region (2014 - 2019)', fontsize = 15)

plt.legend(loc='upper right')



plt.show()

#extracting causes of death (some entries have several causes of death)



#import regular expression library

import re



#dict of given causes of death and frequency

#use regular expression to split string and ignore whitespace

dict_type_count = {}

for d in mm_data['Cause of Death'].index:

    list_temp = []

    list_temp.extend(re.split(r'[,]\s*',mm_data['Cause of Death'][d])) 

    for i in list_temp:

        if i not in dict_type_count:

            dict_type_count[i] = 1

        else:

            dict_type_count[i] += 1



#converting dictionary into series

death_type_count = pd.Series(dict_type_count)

print(death_type_count)
#combine all values with 'unknown' as one entry with index 'Unknown', and combine any cause of death related to forms of transport  as 'Vehicle Accident' 

dict_repeated = {}

dict_repeated['Unknown'] = 0

dict_repeated['Vehicle Accident'] = 0

reps = pd.Series(death_type_count.index)



list_reps =[]



for i in reps[reps.str.contains(r'[Uu]nknown')]:

    dict_repeated['Unknown'] += death_type_count[i]

    list_reps.append(i)

        

for i in reps[reps.str.contains(r'\b[Tt]ruck\b')]:

    dict_repeated['Vehicle Accident'] += death_type_count[i]

    list_reps.append(i)

    

for i in reps[reps.str.contains(r'[Tt]rain')]:

    if i not in list_reps:

        dict_repeated['Vehicle Accident'] += death_type_count[i]

        list_reps.append(i)

        

for i in reps[reps.str.contains(r'\b[Bb]us\b')]: 

    dict_repeated['Vehicle Accident'] += death_type_count[i]

    list_reps.append(i)



for i in reps[reps.str.contains(r'[Vv]ehicle')]: 

    if i != 'Accident (non-vehicle)':

        dict_repeated['Vehicle Accident'] += death_type_count[i]

        list_reps.append(i)

        

death_type_count.drop(list_reps, inplace=True)

death_type_count = death_type_count.append(pd.Series(dict_repeated))

print('The 10 biggest causes of deaths are: \n{} '.format(death_type_count.sort_values(ascending=False)[:10]))
from wordcloud import WordCloud, STOPWORDS





wordcloud = WordCloud(width = 3000, height = 2000 , background_color = 'black', colormap = 'Reds',

                       stopwords = STOPWORDS).generate_from_frequencies(death_type_count.to_dict())



fig = plt.figure(figsize = (30,25))

plt.imshow(wordcloud)

plt.axis('off')



plt.show()

mm_copy = mm_data.copy()

mm_copy.loc[:, 'Number of Incidents'] = 1

mm_copy.head()
season_pattern = mm_copy.pivot_table(index='Reported Month', values='Number of Incidents', aggfunc='sum') 

season_pattern.reset_index(inplace=True)

#season_pattern
dict_months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

season_pattern['Reported Month'] = season_pattern['Reported Month'].map(dict_months)
season_pattern.sort_values(by='Reported Month', inplace=True)

#check

#season_pattern
#plotting number of incidents by month

import calendar

plt.figure(figsize=(10,10))



plt.plot(season_pattern['Reported Month'], season_pattern['Number of Incidents'], 'b-', marker = 'o')

plt.ylim(0, 700)

plt.xlabel('Month',fontsize = 13)

plt.xticks(np.arange(1,13), calendar.month_name[1:13], rotation=20)

plt.ylabel('Number of Incidents', fontsize = 13)

plt.title('Number of Incidents by Month', fontsize = 15)

plt.show()