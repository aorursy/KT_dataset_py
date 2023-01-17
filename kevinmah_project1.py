import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import folium



from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor
countryinfo = pd.read_csv('../input/covid19-useful-features-by-country/Countries_usefulFeatures.csv', engine='python')

covidinfo = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covidinfo.drop(['Province/State', 'Last Update'], axis=1,inplace=True)

countryinfo.at[90, 'Country_Region'] = 'South Korea'

covidinfo_recent = covidinfo[covidinfo['ObservationDate'] == '07/31/2020']

country_covid = covidinfo_recent.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].apply(lambda x : x.astype(int).sum())

country_covid.reset_index(level=0, inplace=True)

country_covid.rename(columns = {'Country/Region':'Country_Region'}, inplace = True)

country_covid.at[176, 'Country_Region'] = 'United Kingdom'

country_covid.at[106, 'Country_Region'] = 'China'

equality_info = pd.read_csv('../input/human-development/inequality_adjusted.csv')



equality_info = equality_info[['Country', 'Human Development Index (HDI)' ]]

equality_info.rename(columns = {'Country':'Country_Region'}, inplace = True)

equality_info.at[7, 'Country_Region'] = 'US'

equality_info.at[50, 'Country_Region'] = 'Russia'

equality_info.at[16, 'Country_Region'] = 'South Korea'

equality_info.at[69, 'Country_Region'] = 'Iran'

equality_info.at[70, 'Country_Region'] = 'Venezuela'

equality_info.at[118, 'Country_Region'] = 'Bolivia'

population_info = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')

population_info = population_info[population_info['alpha3code'].notna()]

population_info.rename(columns = {'country':'Country_Region'}, inplace = True)

population_info_add = population_info.copy()

population_info_add = population_info_add[['Country_Region', 'density', 'urbanpop', 'smokers', 'avgtemp', 'avghumidity']]

cluster_data = countryinfo.copy()

cluster_data['tourism/population'] = cluster_data['Tourism']/cluster_data['Population_Size']

cluster_data = cluster_data[['Country_Region', 'Population_Size', 'Tourism', 'Mean_Age', 'tourism/population' ]]

cluster_data = pd.merge(cluster_data,equality_info,on='Country_Region',how="left")

cluster_data = pd.merge(cluster_data,population_info_add,on='Country_Region',how="left")

cluster_data = cluster_data.dropna()



cluster_data1 = cluster_data.copy()

cluster_data1.drop(['Country_Region'], axis=1,inplace=True)
# Source code from https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f

Sum_of_squared_distances = []

K = range(1,50)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(cluster_data1)

    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
km = KMeans(n_clusters=7)

km = km.fit(cluster_data1)
cluster_data['cluster'] = km.labels_

cluster_data =  pd.merge(cluster_data, country_covid,on='Country_Region',how="left")

cluster_graph = covidinfo.copy()

cluster_graph.rename(columns = {'Country/Region':'Country_Region'}, inplace = True)

cluster_graph1 = cluster_graph.groupby(['Country_Region','ObservationDate'])['Confirmed', 'Deaths', 'Recovered'].apply(lambda x : x.astype(int).sum())

cluster_graph1.reset_index(inplace=True)

cluster_values = cluster_data[['Country_Region', 'cluster']]

cluster_graph1 = pd.merge(cluster_graph1, cluster_values, on='Country_Region',how="left")

cluster_graph1 = cluster_graph1[cluster_graph1['cluster'].notna()]

for i in cluster_graph1.cluster.unique():

    test_df2= cluster_graph1[["Country_Region","ObservationDate","Confirmed","cluster"]]

    test_df2=test_df2[ (test_df2.cluster==i) & (test_df2.ObservationDate > "03/01/2020") ]

    fig = plotly.line(test_df2, x="ObservationDate", y="Confirmed", color='Country_Region')

    fig.show()
feature_data = cluster_data.copy()

feature_data = feature_data.dropna()

y = (feature_data.Confirmed/feature_data.Population_Size)*100000

x = feature_data.drop(['Country_Region', 'cluster', 'Deaths', 'Recovered', 'Confirmed', 'Population_Size', 'Tourism'], axis=1)

clf = RandomForestRegressor(n_estimators=1000)

model = clf.fit(x, y)

score = clf.score(x, y)

print("Accuracy of the model: ", score)

df_result = pd.DataFrame()

df_result['features'] = x.columns

df_result['importance'] = model.feature_importances_

df_result.plot('features', 'importance', 'barh', figsize=(15,8), title='Confirmed Cases')
y = (feature_data.Deaths/feature_data.Population_Size)

model = clf.fit(x, y)

df_result = pd.DataFrame()

df_result['features'] = x.columns

df_result['importance'] = model.feature_importances_

df_result.plot('features', 'importance', 'barh', figsize=(15,8), title='Confirmed Deaths Rate')
y = (feature_data.Recovered/feature_data.Confirmed)

model = clf.fit(x, y)

df_result = pd.DataFrame()

df_result['features'] = x.columns

df_result['importance'] = model.feature_importances_

df_result.plot('features', 'importance', 'barh', figsize=(15,8), title='Percent Cases Recovered')
country_loc = countryinfo[['Country_Region', 'Latitude', 'Longtitude', 'Country_Code']]

map_data = pd.merge(cluster_data, country_loc, on='Country_Region',how="left")

map_data['Confirmed Rate'] = (map_data.Confirmed/map_data.Population_Size)*100000

map_data['Death Rate'] = (map_data.Deaths/map_data.Population_Size)*100000

map_data['Recovery Rate'] = map_data.Recovered/map_data.Confirmed

map_data = map_data.dropna()

fig = plotly.scatter(map_data[(map_data.urbanpop>40) & (map_data['tourism/population'] < 30)], x='urbanpop', y='Confirmed Rate', 

                     color='Country_Region', size='tourism/population', height=600, text='Country_Region',log_x=True,log_y=True, 

                     title="Covid19 Cases per 100000 vs % Population Living in Urban Areas")

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.show()
country_geo = "../input/world-countries/world-countries.json"

m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=8, zoom_start=1.5)

title_html = '''

             <h3 align="center" style="font-size:20px"><b>Covid19 Cases per 100000 vs Urban Center Living</b></h3>

             '''

m.get_root().html.add_child(folium.Element(title_html))



m.choropleth(geo_data=country_geo, data=map_data,

             columns=['Country_Code', 'urbanpop'],

             key_on='feature.id',

             # 'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu','YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.

             fill_color='GnBu', fill_opacity=0.7, line_opacity=0.1,

             legend_name="% Population That Live In Urban Centers")



for i in range(0, len(map_data)):

    folium.Circle(

        location=[map_data.iloc[i]['Latitude'], map_data.iloc[i]['Longtitude']],

        color='crimson', fill='crimson',

        tooltip =   '<li><bold>Country : '+str(map_data.iloc[i]['Country_Region'])+

                    '<li><bold>Confirmed : '+str(map_data.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(map_data.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(map_data.iloc[i]['Recovered'])+

                    '<li><bold>Confirmed Rate : '+str(map_data.iloc[i]['Confirmed Rate'])+

                    '<li><bold>% Population That Live In Urban Centers : '+str(map_data.iloc[i]['urbanpop'])+

                    '<li><bold>Tourism/Population Ratio : '+str(map_data.iloc[i]['tourism/population'])+

                    '<li><bold>% Population that Smoke : '+str(map_data.iloc[i]['smokers'])

        ,

        radius=int(map_data.iloc[i]['Confirmed Rate']*200)).add_to(m)



m
fig = plotly.scatter(map_data[(map_data['Human Development Index (HDI)'] > 0.4)], x='Human Development Index (HDI)', y='Death Rate', color='Country_Region', size='urbanpop', height=600,

                 text='Country_Region',log_x=True,log_y=True, title="Covid19 Deaths per 100000 vs Human Development Index")

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.show()
m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=8, zoom_start=1.5)



title_html = '''

             <h3 align="center" style="font-size:20px"><b>Covid19 Deaths per 100000 vs Human Development Index</b></h3>

             '''

m.get_root().html.add_child(folium.Element(title_html))



m.choropleth(geo_data=country_geo, data=map_data,

             columns=['Country_Code', 'Human Development Index (HDI)'],

             key_on='feature.id',

             # 'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu','YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.

             fill_color='GnBu', fill_opacity=0.7, line_opacity=0.1,

             legend_name="Human Development Index (HDI)")



for i in range(0, len(map_data)):

    folium.Circle(

        location=[map_data.iloc[i]['Latitude'], map_data.iloc[i]['Longtitude']],

        color='crimson', fill='crimson',

        tooltip =   '<li><bold>Country : '+str(map_data.iloc[i]['Country_Region'])+

                    '<li><bold>Confirmed : '+str(map_data.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(map_data.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(map_data.iloc[i]['Recovered'])+

                    '<li><bold>Death Rate : '+str(map_data.iloc[i]['Death Rate'])+

                    '<li><bold>Human Development Index : '+str(map_data.iloc[i]['Human Development Index (HDI)'])+

                    '<li><bold>% Population That Live In Urban Centers : '+str(map_data.iloc[i]['urbanpop'])+

                    '<li><bold>Tourism/Population Ratio : '+str(map_data.iloc[i]['tourism/population'])

        ,

        radius=int(map_data.iloc[i]['Death Rate']*4000)).add_to(m)



m
fig = plotly.scatter(map_data[(map_data['Recovery Rate'] > 0.5) & (map_data.smokers > 8) & (map_data['tourism/population'] < 30)], 

                     x='smokers', y='Recovery Rate', color='Country_Region', size='tourism/population', height=600,

                     text='Country_Region',log_x=True,log_y=True, title="Percent Recovered vs Percent of Population that Smokes")

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.show()
m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=8, zoom_start=1.5)



title_html = '''

             <h3 align="center" style="font-size:20px"><b>Percent Recovered vs Percent of Population that Smokes</b></h3>

             '''

m.get_root().html.add_child(folium.Element(title_html))



m.choropleth(geo_data=country_geo, data=map_data,

             columns=['Country_Code', 'smokers'],

             key_on='feature.id',

             # 'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu','YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.

             fill_color='GnBu', fill_opacity=0.7, line_opacity=0.1,

             legend_name="% Population that smokes")



for i in range(0, len(map_data)):

    folium.Circle(

        location=[map_data.iloc[i]['Latitude'], map_data.iloc[i]['Longtitude']],

        color='crimson', fill='crimson',

        tooltip =   '<li><bold>Country : '+str(map_data.iloc[i]['Country_Region'])+

                    '<li><bold>Confirmed : '+str(map_data.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(map_data.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(map_data.iloc[i]['Recovered'])+

                    '<li><bold>Recovery Rate : '+str(map_data.iloc[i]['Recovery Rate'])+

                    '<li><bold>% Population that Smoke: '+str(map_data.iloc[i]['smokers'])+

                    '<li><bold>Tourism/Population Ratio : '+str(map_data.iloc[i]['tourism/population'])+

                    '<li><bold>Human Development Index : '+str(map_data.iloc[i]['Human Development Index (HDI)'])

        ,

        radius=int(map_data.iloc[i]['Recovery Rate']*100000)).add_to(m)



m