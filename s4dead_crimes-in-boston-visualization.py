import datetime



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium import plugins

import plotly.express as px



from sklearn.neighbors import KNeighborsClassifier



from wordcloud import WordCloud



!pip install alphashape

import alphashape



%matplotlib inline

sns.set()
dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



data = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1',

                   parse_dates=['OCCURRED_ON_DATE'], date_parser=dateparse)
data.head(5)
data.info()
data.isna().sum()
data.describe()
for column in data:

    print(f'{column}: {data[column].unique().size}')
for column in ['DISTRICT', 'SHOOTING', 'YEAR', 'UCR_PART']:

    print(f'{column}: {data[column].unique()}')
data.drop(['INCIDENT_NUMBER', 'OFFENSE_CODE', 'OFFENSE_DESCRIPTION', 'Location'], axis=1, inplace=True)
rename = {'OFFENSE_CODE_GROUP': 'Group',

          'DISTRICT': 'District',

          'REPORTING_AREA': 'Area',

          'SHOOTING': 'If_shooting',

          'OCCURRED_ON_DATE': 'Date',

          'YEAR': 'Year',

          'MONTH': 'YMonth',

          'DAY_OF_WEEK': 'WDay',

          'HOUR': 'DHour',

          'UCR_PART': 'UCR_part',

          'STREET': 'Street',

          'Long': 'Lon'}



data.rename(index=str, columns=rename, inplace=True)
data.WDay = pd.Categorical(data.WDay,

                           categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

                           ordered=True)
data.Lat.replace(-1, None, inplace=True)

data.Lon.replace(-1, None, inplace=True)
data['YDay'] = data['Date'].dt.dayofyear

data['Mday'] = data['Date'].dt.day
data.head(5)
data.columns
data_counties = data[['District', 'Lat', 'Lon']].dropna()
neigh = KNeighborsClassifier(n_neighbors=500, n_jobs=-1)

neigh.fit(data_counties[['Lat', 'Lon']], data_counties['District'])
data_counties['District'] = neigh.predict(data_counties[['Lat', 'Lon']])
district_groups = data_counties.groupby(['District'])

geojson = {'type': 'FeatureCollection'}

geojson['features'] = []



for district, data_district in dict(list(district_groups)).items():

    hull_curr = list(alphashape.alphashape(data_district[['Lon', 'Lat']].values,

                                           alpha=np.sqrt(data_district.shape[0]) * 1.5).exterior.coords)

    geojson['features'].append({'type': 'Feature',

                                'geometry': {

                                    'type': 'Polygon',

                                    'coordinates': [hull_curr]

                                },

                                'properties': {'district': district}})
fig = px.choropleth_mapbox(data_counties, geojson=geojson, color='District',

                           locations='District', featureidkey='properties.district',

                           center={'lat': 42.315, 'lon': -71.1},

                           mapbox_style='carto-positron', zoom=10.5,

                           opacity=0.5)

fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

fig.show()
# Отрисовка гистограмм

def bar_chart(x_vals, y_vals, title=None, x_label=None, y_label=None, if_plot_vals=False):

    n = len(x_vals)

    x_pos = np.arange(n)



    plt.figure(figsize=(12, 8))

    plt.bar(x_pos, y_vals, align='center', alpha=0.6)

    plt.xticks(x_pos, x_vals)

    if title:

        plt.title(title)

    if x_label:

        plt.xlabel(x_label)

    if y_label:

        plt.ylabel(y_label)



    if if_plot_vals:

        for pos, val in zip(x_pos, y_vals):

            plt.text(pos, val, val, ha='center')



    plt.show()
data_year = data.groupby(['Year']).size().reset_index(name='Counts')



bar_chart(data_year.Year, data_year.Counts, 'All crimes each year', 'Year', 'Counts')
data_2016 = data[data['Year'] == 2016]
data_month = data_2016.groupby(['YMonth']).size().reset_index(name='Counts')

data_month.YMonth.replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

                          ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],

                          inplace = True)



bar_chart(data_month.YMonth, data_month.Counts, 'Crimes each month (2016)', 'Month', 'Counts')
data_yday = data_2016.groupby(['YDay']).size().reset_index(name='Counts')



fig, ax = plt.subplots(figsize=(12, 8))

sns.lineplot(x='YDay',

             y='Counts',

             ax=ax,

             data=data_yday)

plt.title('Crimes each day (2016)')

plt.xlabel('Day')
data_wday = data_2016.groupby(['WDay']).size().reset_index(name='Counts')



bar_chart(data_wday.WDay, data_wday.Counts, 'Crimes each week day (2016)', 'Day', 'Count')
data_hour = data_2016.groupby(['DHour']).size().reset_index(name='Counts')



bar_chart(data_hour.DHour, data_hour.Counts, 'Crimes each hour (2016)', 'Hour', 'Count')
sns.catplot(y='UCR_part',

            kind='count',

            height=7,

            aspect=1.5,

            order=data_2016.UCR_part.value_counts().index,

            data=data_2016)
sns.catplot(y='District',

            kind='count',

            height=8,

            aspect=1.5,

            order=data_2016.District.value_counts().index,

            data=data_2016)
data_wc = data_2016.Group.apply(lambda x: x.replace(' ', ''))

text = ' '.join(data_wc)



wordcloud = WordCloud(collocations=False, width=1600, height=800, max_font_size=300,

                      background_color='white', random_state=5).generate(text)



plt.figure(figsize=(20, 17))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
map_crimes = folium.Map(location=[42.315, -71.1], zoom_start=12, tiles='Stamen Toner')

data_heat = data_2016[['Lat', 'Lon']].dropna().values

plugins.HeatMap(data_heat, radius=10).add_to(map_crimes)

map_crimes
map_crimes = folium.Map(location=[42.315, -71.1], zoom_start=12)

data_sample = data_2016[['District', 'Lat', 'Lon']].dropna().sample(10000)

map_dict = {'A1': 'red', 'C11': 'blue', 'E13': 'green', 'B2': 'gray',

            'D14': 'purple', 'C6': 'pink', 'A7': 'cadetblue', 'E5': 'orange',

            'D4': 'darkred', 'B3': 'lightgreen', 'E18': 'darkblue', 'A15': 'darkgreen'}

data_sample.District = data_sample.District.map(map_dict)

for lat, long, target in zip(data_sample.Lat, data_sample.Lon, data_sample.District):

    folium.Circle((lat, long),

                   radius=5,

                   color=target,

                   fill_color='#3186cc').add_to(map_crimes)

map_crimes
data_county = data_2016.groupby(['District']).size().reset_index(name='Counts')



fig = px.choropleth_mapbox(data_county, geojson=geojson, color='Counts',

                           locations='District', featureidkey='properties.district',

                           color_continuous_scale="Viridis", range_color=(0, 20000),

                           center={'lat': 42.315, 'lon': -71.1},

                           mapbox_style="carto-positron", zoom=10.5,

                           opacity=0.5, labels={'Area': 'Crimes number'})

fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

fig.show()
map_crimes = folium.Map(location=[42.315, -71.1], zoom_start=12, min_zoom=12)

data_heat_time = data_2016[['Lat', 'Lon', 'YMonth']].dropna()

data_heat_time = [data_heat_time[data_heat_time['YMonth'] == i][['Lat', 'Lon']].values.tolist() for i in range(1, 13)]

plugins.HeatMapWithTime(data_heat_time, radius=4, auto_play=False, max_opacity=0.8).add_to(map_crimes)

map_crimes
map_crimes = folium.Map(location=[42.315, -71.1], zoom_start=12, tiles='Stamen Terrain')

data_sample = data_2016[['Group', 'Lat', 'Lon', 'District']].dropna().sample(800)

map_dict = {'A1': 'red', 'C11': 'blue', 'E13': 'green', 'B2': 'gray',

            'D14': 'purple', 'C6': 'pink', 'A7': 'cadetblue', 'E5': 'orange',

            'D4': 'darkred', 'B3': 'lightgreen', 'E18': 'darkblue', 'A15': 'darkgreen'}

data_sample.District = data_sample.District.map(map_dict)

for index, row in data_sample.iterrows():

    folium.Marker(row[['Lat', 'Lon']],

                  popup=row['Group'],

                  icon=folium.Icon(color=row['District'])).add_to(map_crimes)

map_crimes