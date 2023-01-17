import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import plotly.graph_objects as go
import plotly.express as px
with open("/kaggle/input/country-data/country_data.json") as response: 
    counties = json.load(response)


df1 = pd.read_csv("/kaggle/input/lwv-oct-2017/LWV_2016-08.csv",            
                   dtype={"fips": str})
df2 = pd.read_csv("/kaggle/input/lwv-oct-2017/LWV_2016-10.csv",
                 dtype={"fips": str})
df3 = pd.read_csv("/kaggle/input/lwv-oct-2017/LWV_2017-10.csv",
                   dtype={"fips": str})
df4 = pd.read_csv("/kaggle/input/lwv-oct-2017/LWV_2019-11.csv",
                   dtype={"fips": str})
df5 = pd.read_csv("/kaggle/input/lwv-oct-2017/LWV_2020-03.csv",
                   dtype={"fips": str})
df6 = pd.read_csv("/kaggle/input/lwv-oct-2017/LWV_2020-07.csv",
                   dtype={"fips": str})

dfus=pd.read_csv("/kaggle/input/simple-map-us-cities-data/uscities.csv")
dfin = pd.read_csv("/kaggle/input/tx-county-income/Untitled spreadsheet - TX Counties with FIPS.csv")

# for plotting major texas cities
mask = dfus["state_id"] == "TX"
texas_cities = dfus[mask]
mask2 = texas_cities["population"] > 200000
big_texas_cities = texas_cities[mask2]
cities_for_map = big_texas_cities[["city", "lat", "lng", "population"]]

# show counties with percent score
fig = px.choropleth(df3, geojson=counties, locations='fips', color='perc_calc_num',
                         color_continuous_scale="Viridis",
                         range_color=(-1, 100),
                         scope="usa",
                         custom_data=["overall_evaluation"],
                         labels={'perc_calc_num':'Website Score', 'fips':'County ID'},
                         hover_name = 'county_name',
                         hover_data=['overall_evaluation']
                          )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# to show texas cities on map by 100,000s
fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lon = cities_for_map['lng'],
    lat = cities_for_map['lat'],
    hoverinfo = 'text',
    text = cities_for_map['city'],
    name = "Major Cities",
    mode = 'markers',
    marker = dict(
        size = cities_for_map["population"]/100000,
        color = 'rgb(102,102,102)', 
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))
# show income by $5,000s
fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lon = cities_for_map['lng'],
    lat = cities_for_map['lat'],
    hoverinfo = 'text',
    text = cities_for_map['city'],
    name = "Income by 5,000",
    mode = 'markers',
    marker = dict(
        size = dfin["Dollars_2018"]/5000,
        color = 'rgb(205,205,102)',
        #colorbar=dict(
         #       title = 'Income in 5,000s',
          #      titleside = 'bottom',
           #     tickmode = 'array',x=0),
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))
fig.update_geos(fitbounds="locations")
fig.layout.update(showlegend=False)
fig.show()
# new dataframe with the all the shared columns
df_all = df1[['county_name', 'fips','date','perc_calc_na', 'perc_calc_num', 'overall_evaluation']]
df_all = df_all.append(df2[['county_name', 'fips','date','perc_calc_na', 'perc_calc_num', 'overall_evaluation']])
df_all = df_all.append(df3[['county_name', 'fips','date','perc_calc_na', 'perc_calc_num', 'overall_evaluation']])
df_all = df_all.append(df4[['county_name', 'fips','date','perc_calc_na', 'perc_calc_num', 'overall_evaluation']])
df_all = df_all.append(df5[['county_name', 'fips','date','perc_calc_na', 'perc_calc_num', 'overall_evaluation']])

# county by percent of scores, animated by date
fig = px.choropleth(df_all, geojson=counties, locations='fips', color='perc_calc_na',
                           color_continuous_scale="Viridis",
                           range_color=(0, 100),
                           scope="usa",
                          labels={'fips':'County ID', "perc_calc_na":"Website Score", "date":"Date", "overall_evaluation":"Overall Evaluation"},
                          hover_name = 'county_name',
                          hover_data=['overall_evaluation'] 
                          ,animation_frame='date') 

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
df = pd.read_csv("/kaggle/input/lwv-oct-2017/LWV_2016-08.csv",
                   dtype={"fips": str})
dfus=pd.read_csv("/kaggle/input/simple-map-us-cities-data/uscities.csv")
dfin = pd.read_csv("/kaggle/input/tx-county-income/Untitled spreadsheet - TX Counties with FIPS.csv")
dfc = pd.read_csv("/kaggle/input/centroids/Texas_Counties_Centroid_Map.csv")

# counties ranked by income
fig = px.choropleth(dfin, geojson=counties, locations='FIPS', color='Rank_in_state_dollars_2018',
                           color_continuous_scale="Viridis",
                           range_color=(1, 254),
                           scope="usa",
                            labels={'Rank_in_state_dollars_2018':"Income Ranking by County"}
                          )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# join score and FIPS to the centroid data
df[['fips']] = df[['fips']].astype(int)
dfc = dfc.merge(df[['fips','perc_calc_num']], how = "left", left_on = "FIPS", right_on = "fips")

# Add centroids to show score
fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lat = dfc['Y (Long)'],
    lon = dfc['X (Lat)'],
    hoverinfo = 'text',
    text = dfc['CNTY_NM'],
    name = "Score",
    mode = 'markers',
    marker = dict(
        size = np.sqrt(dfc["perc_calc_num"]+1),
        color = dfc["perc_calc_num"],
        colorscale="inferno",
        colorbar=dict(
                title = 'Website Rating %',
                titleside = 'top',
                tickmode = 'array',x=-.1), # the x puts the colorbar on the left
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 68, 0)'
        )
    )))
fig.update_geos(fitbounds="locations")
fig.show()