import pandas as pd

import folium

import pycountry

import matplotlib.pyplot as plt







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/mgkt-dataset/data.csv', sep=',')

print(data.columns)

data.head(10)
data['Q2S'].count()
data['Q1S'].describe()

#stats about right answers for the Question 1
#columns with exact answers (QxA) are not needed, 

#we have correct answers number for each question withn QxS. 

#We also remove QxI(question order in which it was displayed for the user) 

#and some other irrelevant columns (screen width, time spent oparticular pages, etc.)



cleaned = data.drop(columns=[("Q"+ str(i)+"A") for i in range(1,33)])

cleaned2 = cleaned.drop(columns=[("Q"+ str(i)+"I") for i in range(1,33)])

cleaned3 = cleaned2.drop(columns=["screenw","screenh","introelapse","testelapse","surveyelapse"])





cleaned3.head(5)
#We need to make values more readable, so inplace replacing is a good option



cleaned3.gender.replace([1, 2, 3, 0], ["male", "female", "other", "other"], inplace=True)

cleaned3.engnat.replace([1, 2], ["yes", "no"], inplace=True)

cleaned3.head(5)
def relative_values(x):

    y = (x +5)*10

    return y



def to_seconds(x):

    y = round(x/1000,1)

    return y
relative_values_cleaned = cleaned3



#to make analysis is cleaner and more representative let's transform 'correct' answers for 

#each question to relative (from -5 to 5, totally 10 options) 



for i in range(1,33):

    clmn = "Q"+str(i)+"S"

    relative_values_cleaned[clmn] = relative_values_cleaned[clmn].apply(relative_values)

relative_values_cleaned.head(5)
#also let's change time spent for each questions to be represented in seconds, 

#not milliseconds. Plus let's round it as well.



for i in range(1,33):

    clmn = "Q"+str(i)+"E"

    relative_values_cleaned[clmn] = relative_values_cleaned[clmn].apply(to_seconds)



relative_values_cleaned.head(5)
#Now we compute average correct score per user

relative_values_cleaned['QxS_avg'] = relative_values_cleaned[[("Q"+ str(i)+"S") for i in range(1,33)]].mean(axis=1)

#Now we compute average time spent for the whole questions answered per user

relative_values_cleaned['QxE_avg'] = relative_values_cleaned[[("Q"+ str(i)+"E") for i in range(1,33)]].mean(axis=1)

#function to convert 2-letter country codes to 3-letter ones to be used for Folium module



def country3(x):

    try:

        return pycountry.countries.lookup(x).alpha_3

    except Exception:

        pass
relative_values_cleaned["country_code"]=relative_values_cleaned["country"].apply(country3)
#creating new DF with columns we want to analyse 

aggregated_stats= relative_values_cleaned[['age','gender', 'engnat', 'country_code', 'QxS_avg', 'QxE_avg']]
aggregated_stats.head(5)
#We need to add new column with the number of participants who took the test per country



aggregated_stats['participants'] = aggregated_stats.groupby(['country_code'])['QxS_avg'].transform('count')

aggregated_stats.head(5)
aggregated_stats[['participants', 'country_code']].groupby('country_code', as_index=False)['participants'].mean().sort_values(by='participants', ascending=False)[:10].plot(kind='bar',title="Participants by country (most 10)", x='country_code',y='participants',figsize=(15,8))

plt.show()
filter1 = aggregated_stats['participants'] >=100



aggregated_stats[filter1][['QxS_avg', 'country_code']].groupby('country_code', as_index=False)['QxS_avg'].mean().sort_values(by='QxS_avg', ascending=False)[:10].plot(kind='bar',ylim=70,title="Average scores by country (most 10) with more than 100 participants", x='country_code',y='QxS_avg',figsize=(15,8))

plt.show()
filter1 = aggregated_stats['engnat'] == "no"

filter2 = aggregated_stats['participants'] >=10



aggregated_stats[filter1][filter2][['QxS_avg', 'country_code']].groupby('country_code', as_index=False)['QxS_avg'].mean().sort_values(by='QxS_avg', ascending=False)[:10].plot(kind='bar',ylim=75,title="Average scores by country (most 10) with more than 10 participants, non-native EN speakers", x='country_code',y='QxS_avg',figsize=(15,8))

plt.show()
filter1 = aggregated_stats['age'] <= 85

aggregated_stats[filter1][['QxS_avg', 'age']].groupby('age', as_index=False)['QxS_avg'].mean().sort_values(by='age', ascending=False).plot(kind='line',title="Average scores by age of participants", x='age',y='QxS_avg',figsize=(15,8), grid='true')

plt.show()
filter1 = aggregated_stats['age'] <= 85

filter_male = aggregated_stats['gender'] =='male'

filter_female = aggregated_stats['gender'] =='female'





ax = plt.gca()



aggregated_stats[filter1][filter_male][['QxS_avg', 'age']].groupby('age', as_index=False)['QxS_avg'].mean().sort_values(by='age', ascending=False).plot(kind='line',title="Average scores by age for male and females", x='age',y='QxS_avg',figsize=(15,8), grid='true', ax=ax)

aggregated_stats[filter1][filter_female][['QxS_avg', 'age']].groupby('age', as_index=False)['QxS_avg'].mean().sort_values(by='age', ascending=False).plot(kind='line', x='age',y='QxS_avg',figsize=(15,8), grid='true',color='red',ax=ax)

plt.show()
#Let's check what countries perform better taking into account their native language is NOT English

value_to_map = "QxS_avg"

filter1 = aggregated_stats['engnat']== "no"

filter2 = aggregated_stats['participants'] >=10





aggregated_stats_with_filters = aggregated_stats[filter1][filter2]



country_geo = '/kaggle/input/python-folio-country-boundaries/world-countries.json'

plot_data = aggregated_stats_with_filters[["country_code",value_to_map]]

map = folium.Map(location=[0, 0], zoom_start=2.0)

map.choropleth(geo_data=country_geo, data=plot_data,

             columns=["country_code", value_to_map],

             key_on='feature.id',

             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,

             legend_name="Average rating (correct answers %) by country with filters applied")

map.save('plot_data1.html')



from IPython.display import HTML

HTML('<iframe src=plot_data1.html width=900 height=450></iframe>')

#Countries with the highest rating AND native English speakers

value_to_map = "QxS_avg"

filter1 = aggregated_stats['participants']>=100



aggregated_stats_with_filters = aggregated_stats[filter1]



country_geo = '/kaggle/input/python-folio-country-boundaries/world-countries.json'

plot_data = aggregated_stats_with_filters[["country_code",value_to_map]]

map = folium.Map(location=[0, 0], zoom_start=2.0)

map.choropleth(geo_data=country_geo, data=plot_data,

             columns=["country_code", value_to_map],

             key_on='feature.id',

             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,

             legend_name="Average rating (correct answers %) by country with filters applied")

map.save('plot_data2.html')



from IPython.display import HTML

HTML('<iframe src=plot_data2.html width=900 height=450></iframe>')