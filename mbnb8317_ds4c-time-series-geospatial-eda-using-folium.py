import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import matplotlib.pyplot as plt

import folium 

from folium import plugins



import json 

from datetime import datetime



import warnings

warnings.filterwarnings("ignore")
patient = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')

route = pd.read_csv('/kaggle/input/coronavirusdataset/PatientRoute.csv')
patient.head()
route.head()
print('There is some difference between Male and Female.')

print('But, it is difficult to be considered that Female is more vulnerable to CVD-19 than male.')

sns.countplot(patient['sex'])

plt.title('Distribution of Sex')

plt.show()
patient['age'] = 2020 - patient['birth_year'] + 1

patient['age_group'] = patient['age'] // 10

patient['age_group'] = [str(a).replace('.','') for a in patient['age_group']]



print('Female 20, 40, 50 age rank top with significant difference.')

print('What happens to them? It might be a next topic of my analysis')

plt.figure(figsize = (15,8))

ax = sns.countplot(patient['sex'].str.cat(patient['age_group']))

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
sns.countplot(patient['country'])

plt.title('Distribution of Country')

plt.show()
ax = sns.countplot(patient['infection_case'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('Distribution of infection_case')

plt.show()
print('As you can see Capital area, Daegu, and Gyeongsangbuk-do rank top 3.')

ax = sns.countplot(patient['province'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('How many patient by each province')

plt.show()
ax = sns.countplot(patient['infection_order'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('Distribution of Infection order')

plt.show()
print('In my opinion, there seems to be 7 anomalies points, which locate above the red line.')

print('They might have a important role on spread of CVD 19.')

print('Lets take a look at those.')

sns.scatterplot(patient['patient_id'],patient['contact_number'])

plt.plot([0,7000000000], [200,200], color = 'red')

plt.title('Distribution of Contact_number')

plt.show()
anomaly_id = patient.query('contact_number > 200')['patient_id'].values; display(anomaly_id)
def make_polygon(route_df, patient_df, id_num):

    

    id_filter = route_df.query('patient_id == @id_num')

    id_filter['timestamp'] = [json.dumps(date+'T00:00:00') for date in id_filter['date']]

    

    ctc = int(patient_df.query('patient_id == @id_num')['contact_number'].values[0])

    

    print(f'{id_num}th patient made contacts with {ctc} in those path')

    

    id_filter_shift = id_filter.shift(-1) 



    # set the zoom point

    m = folium.Map([37, 128], zoom_start=7)



    # set the icon on Incheon airport and gimpo airport

    icon_plane1 = plugins.BeautifyIcon( icon='plane', border_color='#b3334f', text_color='#b3334f', icon_shape='triangle')

    icon_plane2 = plugins.BeautifyIcon( icon='plane', border_color='#b3334f', text_color='#b3334f', icon_shape='triangle')



    folium.Marker( location=[37.4692, 126.451], popup='incheon airport', icon=icon_plane1).add_to(m)

    folium.Marker( location=[37.558808, 126.794458], popup='gimpo airport', icon=icon_plane2 ).add_to(m)



    # add the fullscreen utility. if you don't need, it's ok to remark the line :)

    plugins.Fullscreen( position='topright', title='Click to expand', title_cancel='Click to exit', force_separate_button=True ).add_to(m)



    folium.Polygon( locations = id_filter[['latitude','longitude']], fill = True, tooltip = 'Polygon' ).add_to(m) 

    

    for lat,lon in zip(id_filter.latitude, id_filter.longitude): 

        folium.Circle( location = [lat,lon], 

                      radius = 400,

                     color = 'red').add_to(m)

    

    return m
make_polygon(route, patient, 1200000031)
make_polygon(route, patient, 1300000001)
make_polygon(route, patient, 2000000003)
make_polygon(route, patient, 2000000006)
def make_polygon_time(route_df, patient_df, id_num):

    print(f'Where {id_num}th patient went for several days')

    

    

    # set the zoom point

    m = folium.Map([37, 128], zoom_start=7)

    

    id_filter = route_df.query('patient_id == @id_num')

    id_filter['timestamp'] = [date+'T00:00:00' for date in id_filter['date']]

    id_filter_shift = id_filter.shift(-1) 

    

    # set the icon on Incheon airport and gimpo airport

    icon_plane1 = plugins.BeautifyIcon( icon='plane', border_color='#b3334f', text_color='#b3334f', icon_shape='triangle')

    icon_plane2 = plugins.BeautifyIcon( icon='plane', border_color='#b3334f', text_color='#b3334f', icon_shape='triangle')



    folium.Marker( location=[37.4692, 126.451], popup='incheon airport', icon=icon_plane1 ).add_to(m)

    folium.Marker( location=[37.558808, 126.794458], popup='gimpo airport', icon=icon_plane2 ).add_to(m)

    

    # add the fullscreen utility. if you don't need, it's ok to remark the line :)

    plugins.Fullscreen( position='topright', title='Click to expand', title_cancel='Click to exit', force_separate_button=True ).add_to(m)



    

    lines = []

    for lon, lat, time, lon_s, lat_s, time_s in zip(id_filter['longitude'], id_filter['latitude'], id_filter['timestamp'], 

                                                    id_filter_shift['longitude'], id_filter_shift['latitude'], id_filter_shift['timestamp']):

        temp_dict = {}

        temp_dict['coordinates'] = [[lon,lat], [lon_s,lat_s]]

        temp_dict['dates'] = [time,time_s]

        temp_dict['color'] = 'red'



        lines += [temp_dict]



    del lines[-1]



    features = [ { 'type': 'Feature', 

                  'geometry': { 'type': 'LineString', 

                               'coordinates': line['coordinates'], }, 

                  'properties': { 'times': line['dates'], 

                                 'style': { 'color': line['color'], 

                                           'weight': line['weight'] if 'weight' in line else 5 } } } for line in lines ] 



    plugins.TimestampedGeoJson({ 'type': 'FeatureCollection', 'features': features, }, period='P1D', add_last_point=True).add_to(m) 



    return m
make_polygon_time(route, patient, 1200000031)
make_polygon_time(route, patient, 1300000001)
make_polygon_time(route, patient, 2000000003)
make_polygon_time(route, patient, 2000000006)