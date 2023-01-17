from IPython.display import Image

Image('../input/header/smile.jpg')
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

import geopandas as gpd

import plotly.express as px

import plotly.graph_objects as go

pd.options.plotting.backend = "plotly"
# Create a dictionary with each data set as items.

years = [2015, 2016, 2017, 2018, 2019, 2020]

dataset = {year : pd.read_csv(f'../input/world-happiness-report/{year}.csv')

              for year in years}
# Column headers are inconsistent between files, so we give them the same names.

for i, item in dataset.items():

    item.rename(columns = {'Ladder score' : 'Score',

                           'GDP per capita' : 'Logged GDP per capita',

                           'Happiness.Rank' : 'Overall rank',

                           'Happiness Rank' : 'Overall rank',

                           'Happiness.Score' : 'Score',

                           'Happiness Score' : 'Score',

                           'Economy..GDP.per.Capita.' : 'Logged GDP per capita',

                          'Health..Life.Expectancy.' : 'Healthy life expectancy',

                          'Trust..Government.Corruption.' : 'Perceptions of corruption',

                          'Family' : 'Social support',

                          'Freedom' : 'Freedom to make life choices',

                           'Dystopia.Residual' : 'Dystopia Residual',

                           'Economy (GDP per Capita)' : 'Logged GDP per capita',

                           'Health (Life Expectancy)' : 'Healthy life expectancy',

                           'Trust (Government Corruption)' : 'Perceptions of corruption',

                           'Country name' : 'Country',

                           'Country or region' : 'Country'

                          }, 

                inplace=True)
# Divide the variables into "explained by" and regular variables.

variables =  ['Explained by: Log GDP per capita', 

             'Explained by: Social support',

             'Explained by: Healthy life expectancy',

             'Explained by: Freedom to make life choices',

             'Explained by: Generosity',

             'Explained by: Perceptions of corruption',

             ]

variables2 =  ['Logged GDP per capita',

             'Social support',

             'Healthy life expectancy',

             'Freedom to make life choices',

             'Generosity',

             'Perceptions of corruption',

             ]
# Make the country names the index for each data frame.

for year in years:

    dataset[year].set_index('Country', inplace=True)
dataset[2020][['Score'] + variables2].style.background_gradient(cmap='Blues')
dataset[2020][['Score'] + variables2].corr().style.background_gradient(cmap='RdBu', vmin=-1, vmax=1)
for var in variables2:

    dataset[2020].plot.scatter(x=var,

                               y='Score',

                              hover_name=dataset[2020].index,

                              trendline="lowess").show()
shapefile = '../input/map-files/ne_110m_admin_0_countries.shp'

#Read shapefile using Geopandas

gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

#Rename columns.

gdf.columns = ['country', 'country_code', 'geometry']

#Drop row corresponding to 'Antarctica'

gdf = gdf.drop(gdf.index[159])
replacements = {'United States of America':'United States',

               'Czechia':'Czech Republic',

                'Taiwan' : 'Taiwan Province of China',

                'Republic of Serbia' : 'Serbia',

                'Palestine' : 'Palestinian Territories',

                'Republic of the Congo' : 'Congo (Kinshasa)',

                'eSwatini' : 'Swaziland',

                'United Republic of Tanzania' : 'Tanzania',

                }
# Make names match

gdf = gdf.replace(replacements)
#Merge dataframes gdf and dataset[2020].

merged = gdf.merge(dataset[2020], left_on = 'country', right_on = 'Country', how='left')
fig = px.choropleth(merged, locations="country_code",

                    color='Score',

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.RdBu,

                   )

fig.update_layout(

    autosize=False,

    width=950,

    height=600,)

fig.show()
happiness_ts = pd.DataFrame()

for year in years:

    happiness_ts = pd.concat([happiness_ts, dataset[year]['Score']], axis=1)



happiness_ts.columns = years
df = dataset[2019].copy()

df['Delta'] = happiness_ts[2019] - happiness_ts[2015]

df['2015 Score'] = happiness_ts[2015]
#Merge dataframes gdf and df_2016.

merged = gdf.merge(df, left_on = 'country', right_on = 'Country', how='left')
fig = px.choropleth(merged, locations="country_code",

                    color='Delta',

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.RdBu,

                   )

fig.update_layout(

    autosize=False,

    width=950,

    height=600,)

fig.show()
px.box(happiness_ts,

       points='all',

       hover_data=[happiness_ts.index],

       labels=dict(variable="Year", value="Score"),

      )
ts = {var : pd.DataFrame(columns=years[:-1]) for var in variables2}

for var in variables2:

    for year in years[:-1]:

        ts[var][year] = dataset[year][var]
for var in variables2:

    fig = px.box(ts[var],

                 points='all',

                 hover_data=[ts[var].index],

                 labels=dict(variable="Year", value=var)

                )

    fig.show()
year = 2020

clusterset = dataset[year][:len(dataset[year])//4]
kdata = clusterset[variables]

X = kdata.values
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, '.')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.grid()

plt.show()
nclusters = 3

kmeans = KMeans(n_clusters = nclusters, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)
clusterset['cluster']= y_kmeans
#Merge dataframes gdf and df_2016.

merged = gdf.merge(clusterset, left_on = 'country', right_on = 'Country', how='left')



# Cluster names need to be strings so we can colour by cluster.

merged['cluster'] = merged['cluster'].astype(str)
fig = px.choropleth(merged, locations="country_code",

                    color="cluster", 

                    hover_name="country", # column to add to hover information

                    color_discrete_map={'0.0':'green',

                                        '1.0':'red',

                                        '2.0':'blue',

                                        'nan': 'gray'},

                   )



fig.show()
# Find median values for each variable for each cluster.

cluster_median = {i : clusterset[clusterset['cluster']==i].median() for i in range(nclusters)}
fig = px.box(clusterset,

                 y='Score',

                 facet_col='cluster',

                 points='all',

                 hover_data=[clusterset.index],

                ) 

fig.show()
fig = go.Figure()

for i in range(nclusters-1, -1, -1):

    fig.add_trace(go.Scatterpolar(r=cluster_median[i][variables], theta=variables, fill='toself', name='Cluster ' + str(i)))

fig.show()
fig = px.box(clusterset,

             y=['Explained by: Log GDP per capita',

                 'Explained by: Social support',

                 'Explained by: Healthy life expectancy',],

             color='cluster',

             points='all',

             hover_data=[clusterset.index],

            ) 

fig.show()
fig = px.box(clusterset,

             y=['Explained by: Freedom to make life choices',

                 'Explained by: Generosity',

                 'Explained by: Perceptions of corruption'],

             color='cluster',

             points='all',

             hover_data=[clusterset.index],

            ) 

fig.show()
nations = ['Singapore', 'Costa Rica']



fig = go.Figure()

for nation in nations:

    fig.add_trace(go.Scatterpolar(r=clusterset.loc[nation][variables],

                                  theta=variables,

                                  fill='toself',

                                  name=nation + ' : ' + str(clusterset.loc[nation]['Score'])

                                 )

                 )

fig.show()
fig = go.Figure()

for region in dataset[2020]['Regional indicator'].unique():

    data = dataset[2020].where(dataset[2020]['Regional indicator'] == region).dropna().median()

    fig.add_trace(go.Scatterpolar(r=data[variables],

                                      theta=variables,

                                      fill='toself',

                                      name=region + ' : ' + str(data['Score'])

                                     )

                     )

fig.show()