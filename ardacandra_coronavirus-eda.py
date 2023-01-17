import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import folium

from folium import Marker, Circle, PolyLine

from folium.plugins import HeatMap, MarkerCluster

import datetime as dt
case_df = pd.read_csv('../input/coronavirusdataset/case.csv')

patient_df = pd.read_csv('../input/coronavirusdataset/patient.csv')

route_df = pd.read_csv('../input/coronavirusdataset/route.csv')

time_df = pd.read_csv('../input/coronavirusdataset/time.csv')

trend_df = pd.read_csv('../input/coronavirusdataset/trend.csv')

case_df.head()
patient_df.head()
route_df.head()
time_df.head()
trend_df.head()
trend_df.shape
trend_df = trend_df.set_index('date')

trend_df.head()
import matplotlib.ticker as ticker



fig, ax = plt.subplots(figsize=(15, 10))

sns.lineplot(data=trend_df, ax=ax)

plt.title('The Trend of Cold, Flu, Pneumonia, and Coronavirus')



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels()):

    item.set_fontsize(20)

for item in ax.get_xticklabels():

    item.set_fontsize(10)    

    

tick_spacing = 5

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.autofmt_xdate()



plt.show()
time_df.shape
time_df.columns
time_df = time_df.set_index('date')

time_df.head()
time_df.describe()
time_df.info()
time_df['time'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))

sns.lineplot(data=time_df[['test', 'negative', 'confirmed', 'released', 'deceased']], ax=ax)

plt.title('The Number of Tests and Results')



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels()):

    item.set_fontsize(20)

for item in ax.get_xticklabels():

    item.set_fontsize(10)    

    

tick_spacing = 5

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.autofmt_xdate()



plt.show()
last_date = max(time_df.index)

latest_numbers = time_df.drop(['time', 'test', 'negative', 'confirmed', 'released', 'deceased'], axis=1).loc[last_date, :]



fig = px.pie(values=latest_numbers.values, names=latest_numbers.index, title='Distribution of Cases in South Korea')

fig.update_traces(textposition='inside')

fig.show()
fig, ax = plt.subplots( figsize=(8, 8))



sns.lineplot(data=time_df[['Daegu', 'Gyeongsangbuk-do', 'Seoul']], ax=ax)

plt.title('Number of Cases in Daegu, Gyeonsangbuk-do, and Seoul')



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels()):

    item.set_fontsize(20)

for item in ax.get_xticklabels():

    item.set_fontsize(10)    

    

tick_spacing = 5

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.autofmt_xdate()



plt.show()
case_df.head()
case_df[['latitude', 'longitude']] = case_df[['latitude', 'longitude']].replace('-', np.nan)
m_1 = folium.Map(location=[37, 126], tiles='openstreetmap', zoom_start=6)



for idx, row in case_df.iterrows():

    if pd.notnull(row['latitude']):

        Marker([row['latitude'], row['longitude']], popup=folium.Popup((

                                                            'Province : {province}<br>'

                                                            'City : {city}<br>'

                                                            'Group : {group}<br>'

                                                            'Infection Case :{case}<br>'

                                                            'Confirmed : {confirmed}').format(

                                                            province=row['province'],

                                                            city=row['city'],

                                                            group=row['group'],

                                                            case=row['infection_case'],

                                                            confirmed=row['confirmed']), max_width=450)

              ).add_to(m_1)



        Circle(location=[row['latitude'], row['longitude']],

               radius=row['confirmed']*5,

               fill=True

              ).add_to(m_1)

    

m_1
patient_df = patient_df.set_index('patient_id')

patient_df.head()
patient_df['age'] = np.subtract(2020, patient_df['birth_year'], dtype=np.int32)



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))

fig.tight_layout()



index = patient_df['sex'].dropna().value_counts().index

values = patient_df['sex'].dropna().value_counts().values

sns.barplot(x=index, y=values, ax=ax1)

ax1.set_title('Patients Gender Distribution')



age_df = patient_df[['sex', 'age']].dropna()

sns.distplot(age_df.loc[age_df['sex']=='male']['age'], hist=True, bins=30, ax=ax2)

ax2.set_title('Male Age Distribution')



sns.distplot(age_df.loc[age_df['sex']=='female']['age'], hist=True, bins=30, ax=ax3)

ax3.set_title('Female Age Distribution')

    

plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

fig.tight_layout()



index = patient_df['region'].value_counts().index

values = patient_df['region'].value_counts().values

sns.barplot(y=index, x=values, ax=ax1)

ax1.set_title('Region')



index = patient_df['group'].value_counts().index

values = patient_df['group'].value_counts().values

sns.barplot(y=index, x=values, ax=ax2)

ax2.set_title('Group')



index = patient_df['infection_reason'].value_counts().index

values = patient_df['infection_reason'].value_counts().values

sns.barplot(y=index, x=values, ax=ax3)

ax3.set_title('Infection Reason')

    

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))

fig.tight_layout()



index = patient_df['state'].value_counts().index

values = patient_df['state'].value_counts().values

sns.barplot(x=index, y=values, ax=ax1)

ax1.set_title('State')



index = patient_df['infected_by'].value_counts().index.astype(int)

values = patient_df['infected_by'].value_counts().values

sns.barplot(x=index, y=values, ax=ax2)

ax2.set_title("Infected by (patient's id)")



sns.distplot(patient_df['contact_number'].dropna(), hist=False, bins=5,ax=ax3).set(xlim=0)

ax3.set_title('Contact Number')



plt.show()
patient_df['confirmed_date'] = pd.to_datetime(patient_df['confirmed_date'])

patient_df['released_date'] = pd.to_datetime(patient_df['released_date'])

patient_df['deceased_date'] = pd.to_datetime(patient_df['deceased_date'])
patient_df['period_before_release'] = patient_df['released_date'] - patient_df['confirmed_date']

patient_df['period_before_death'] = patient_df['deceased_date'] - patient_df['confirmed_date']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

fig.tight_layout()



period_df = patient_df['period_before_release'].dropna().dt.days



sns.distplot(period_df, hist=False, bins=5,ax=ax1).set(xlim=0)

ax1.set_title('Time Between Confirmed and Released (days)')



period_df = patient_df['period_before_death'].dropna().dt.days



sns.distplot(period_df, hist=False, bins=5,ax=ax2).set(xlim=0)

ax2.set_title('Time Between Confirmed and Death (days)')



plt.show()
route_df.head()
m_2 = folium.Map(location=[37, 126], tiles='cartodbpositron', zoom_start=6)



current_id = 1

points = []



for idx, row in route_df.iterrows():

    if pd.notnull(row['latitude']):

        (Marker([row['latitude'], row['longitude']], 

               icon=folium.Icon(color='red'),

               popup=folium.Popup((

                                                            'Patient id : {patient_id}<br>'

                                                            'Date : {date}<br>'

                                                            'Province : {province}<br>'

                                                            'City :{city}<br>'

                                                            'Visit : {visit}').format(

                                                            patient_id=row['patient_id'],

                                                            date=row['date'],

                                                            province=row['province'],

                                                            city=row['city'],

                                                            visit=row['visit']), max_width=450)

              )).add_to(m_2)

        

        if row['patient_id'] == current_id:

            points.append(tuple([row['latitude'], row['longitude']]))

        else :

            PolyLine(points, color='blue').add_to(m_2)

            current_id = row['patient_id']

            points = []

            points.append(tuple([row['latitude'], row['longitude']]))



m_2