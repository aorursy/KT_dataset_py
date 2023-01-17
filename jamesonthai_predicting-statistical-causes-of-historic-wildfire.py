import sqlite3
import pandas as pd

# Graphing Libraries
import plotly
import plotly.graph_objects as go
import plotly.express as px

# getting geojson counties
from urllib.request import urlopen
import json

# Preprocessing libraries
from sklearn import preprocessing
import datetime

# Create your connection.
con = sqlite3.connect('../input/188-million-us-wildfires/FPA_FOD_20170508.sqlite')
res = con.execute("SELECT name FROM sqlite_master WHERE type='table';")

fires_df = pd.read_sql_query("SELECT * FROM Fires", con)
agency = pd.read_sql_query("SELECT * FROM NWCG_UnitIDActive_20170109", con)
# df = pd.read_sql_query("SELECT * FROM idx_Fires_Shape_node", con)

res.close()
con.close()

# Downloading geojson data of Counties
json_fp = open('../input/uscounties/counties.json')
counties = json.load(json_fp)

# Obtaining FIPS codes
state_fips = pd.read_csv("../input/fipsstate/fips_state.csv", dtype=str)

state_fires = fires_df['STATE'].value_counts().rename_axis('STATES').reset_index(name='counts')
fig = go.Figure(data = go.Choropleth(locations = state_fires['STATES'], #Spatial coordinates
                    z = state_fires['counts'], # Data to be color-coded,
                    locationmode = 'USA-states', # Set of locations to match entries in locations
                    colorscale = 'thermal',
                    colorbar_title = "Fire Intensity"
                                    ))
fig.update_layout(
    title_text = '1992-2015 United States Fires',
    geo = dict(
        scope = 'usa',
        showlakes=False
    )
)
fig.show()
del fig
# Fire origins in reported counties
county_fires = fires_df[['FIPS_CODE', 'STATE']]
state_fips = state_fips[["fips_code", "post_code"]].rename(columns={"fips_code" : "state_fips", "post_code" : "STATE"})
county_fires

full_fips = pd.merge(state_fips, county_fires, on ="STATE")
filtered_fips = full_fips.dropna()
filtered_fips["full_fips"] = filtered_fips["state_fips"].astype(str) + filtered_fips["FIPS_CODE"].astype(str)
reported_fires = filtered_fips["full_fips"].value_counts().rename_axis('fips').reset_index(name='counts')
# reported_fires

# Graphing section
fig = px.choropleth(reported_fires, geojson = counties, locations='fips', color='counts',
                   color_continuous_scale="thermal",
                   labels={'counts': 'Fire Intensities'})
fig.update_traces(marker_line_width=0, marker_opacity=0.8)
fig.update_layout(
                  title_text = '1992 - 2015: Summary of Traceable County Fires',
                  geo = dict(
                    scope='usa',
                    showlakes=False,
                    showsubunits=True,
                    subunitcolor='black'
                  )
                )
fig.show()
del fig
none_fips = full_fips[full_fips.isnull().any(axis=1)]["STATE"]
none_counts = none_fips.value_counts().rename_axis('STATES').reset_index(name='counts')
# none_counts
fig = go.Figure(data = go.Choropleth(locations = none_counts['STATES'], #Spatial coordinates
                    z = none_counts['counts'], # Data to be color-coded,
                    locationmode = 'USA-states', # Set of locations to match entries in locations
                    colorscale = 'thermal',
                    colorbar_title = "Fire Intensity"
                                    ))
fig.update_layout(
    title_text = '1992-2015 : Summary of Unknown State-county Fires',
    geo = dict(
        scope = 'usa',
        showlakes=False
    )
)
fig.show()
total_county_fires = reported_fires['counts'].sum()
total_fires = state_fires['counts'].sum()
total_nocounty_fires = none_counts['counts'].sum()
print(total_fires, total_county_fires, total_nocounty_fires)
a = (total_county_fires / total_fires) * 100
b = (total_nocounty_fires / total_fires) * 100
print("Percentage of Total Reported County Fires " + str(round(a)) + "%")
print("Percentage of Total Unknown County Fires " + str(round(b)) + "%")
# Clearing All seen variables
del total_county_fires
del total_fires
del total_nocounty_fires
del a
del b
del none_fips
del none_counts
del county_fires
# del state_fips
del full_fips
del filtered_fips
del reported_fires
del fig
county_year_slider = fires_df[['STATE', 'FIRE_YEAR', 'FIPS_CODE']]

county_year_slider = pd.merge(state_fips, county_year_slider, on ="STATE")
county_year_fips = county_year_slider.dropna()
county_year_fips["full_fips"] = county_year_fips["state_fips"].astype(str) + county_year_fips["FIPS_CODE"].astype(str)
county_year_fips = county_year_fips[["full_fips","FIRE_YEAR"]] \
                .groupby(['full_fips','FIRE_YEAR']).size() \
                .reset_index(name='counts').sort_values(by=['FIRE_YEAR'])
fig = px.choropleth(county_year_fips, geojson = counties, locations='full_fips', color='counts', 
                    animation_frame='FIRE_YEAR', animation_group = 'full_fips',
                   color_continuous_scale="thermal",
                    hover_name = 'full_fips',
                    hover_data = {
                        'full_fips': False,
                        'FIRE_YEAR' : False,
                        'counts' : False,
                        'Fire Intensity' : county_year_fips['counts']
                    },
                   labels={'counts': 'Fire Intensities'}
                   )
fig.update_traces(marker_line_width=0, marker_opacity=0.8)

fig.update_layout(
                  title_text = '1992 - 2015 Traceable County Fires',
                  geo = dict(
                    scope='usa',
                    showlakes=False,
                    showsubunits=True,
                    subunitcolor='black'
                  )
                )
fig.show()
del fig
del county_year_slider
del county_year_fips
wildfire_causes = fires_df[['STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR', 'FIRE_YEAR', 'FIPS_CODE', 'STATE']]
wildfire_causes.head()

dict_values = {1 : "Lightning", 2 : "Equipment Use", 3 : "Smoking", 4: "Campfire", 5 : "Debris Burning", 6: "Railroad"
              , 7: "Arson", 8 : "Children", 9 : "Miscellaneous", 10 : "Fireworks", 11 : "Powerline", 12 : "Structure", 
               13 : "Missing/Undefined"}

uni_values = wildfire_causes[['STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR', 'FIRE_YEAR']] \
            .groupby(['STAT_CAUSE_DESCR', 'STAT_CAUSE_CODE', 'FIRE_YEAR']).size().reset_index(name='counts') \
            .sort_values(by=['STAT_CAUSE_CODE'])

uni_values

# t1 = uni_values.loc(uni_values['FIRE_YEAR'] == 2012)
fig = px.bar(uni_values, x = 'FIRE_YEAR', y = 'counts', color = 'STAT_CAUSE_DESCR')
fig.update_layout(
                  title_text = '1992 - 2015 Statistical Causes of fires',
    
)
               
fig.show()
del fig
# Taking a further look into statistical causes of fires in relation to how many acres burned for each year
wildfire_cause_fire = fires_df[['STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR', 'FIRE_YEAR', 'FIRE_SIZE']]

fire_acres = wildfire_cause_fire[['STAT_CAUSE_DESCR', 'FIRE_YEAR', 'FIRE_SIZE']] \
    .groupby(['STAT_CAUSE_DESCR', 'FIRE_YEAR']).agg(FIRE_SIZE=pd.NamedAgg(column='FIRE_SIZE', aggfunc='sum')).reset_index()
fire_acres
# Bar plot creation modified from towarddatascience bar-chart-race
import plotly.graph_objects as go
initial_year = fire_acres['FIRE_YEAR'].min()
final_year = fire_acres['FIRE_YEAR'].max()
year_frames = []

for year in range(initial_year, final_year):
    year_data = fire_acres.loc[fire_acres['FIRE_YEAR'] == year]
    max_range = year_data['FIRE_SIZE'].max()
    year_frames.append(go.Frame(data= [go.Bar( x = year_data['STAT_CAUSE_DESCR'], 
                y = year_data['FIRE_SIZE'], cliponaxis=False, text = year_data['FIRE_SIZE'], textposition='outside', 
                                              hoverinfo='none', texttemplate= '%{text: .2s}')],
                layout=go.Layout(title= "Wildfire damage: " + str(year),
                    font={'size': 14}, xaxis= {'showline': False, 'visible' : True},
                    yaxis = {'showline' : False, 'visible' : True, 
                             'autorange' : True, 'fixedrange' : False, 'range': (0, max_range + 100000)}, bargap = 0.15
                                )))
init = fire_acres.loc[fire_acres['FIRE_YEAR'] == initial_year]
max_range = fire_acres['FIRE_SIZE'].max()
fig = go.Figure(
    data = [go.Bar(x = init['STAT_CAUSE_DESCR'], y = init['FIRE_SIZE'],
            cliponaxis=False, hoverinfo='none', textposition='outside', text = init['FIRE_SIZE'],
                  texttemplate= '%{text: .2s}')],
    layout=go.Layout(
        yaxis = {'showline' : False, 'visible' : True, 
                'autorange' : True, 'fixedrange' : False, 'range' : (0, max_range + 100000)}, bargap = 0.15,
        xaxis= {'showline': False, 'visible' : True},
        updatemenus=[dict(type="buttons",
                         buttons=[dict(label="Play", method = "animate",
                        args = [None, {"frame" : {"duration": 2000, "redraw" :True}, "fromcurrent": True}]),
                        dict(label="Stop", method = "animate",
                        args = [[None], {"frame": {"duration": 0, "redraw": False}, "mode" : "immediate", "transition": {"duration" : 40}}])])]
    ),
    frames = list(year_frames)
)
fig.update_yaxes(automargin=True)
fig.show()
total_acres = fire_acres[['FIRE_YEAR', 'FIRE_SIZE']].groupby(['FIRE_YEAR']) \
        .agg(FIRE_SIZE=pd.NamedAgg(column='FIRE_SIZE', aggfunc='sum')).reset_index()
fig = px.line(total_acres, x = "FIRE_YEAR", y = "FIRE_SIZE", title = "Acres burned across the US")
fig.show()
fires_df.isna().sum()
# One thing I want to also look at is also look at the unit name and see what report it is from
agency.isna().sum()
full_data = pd.merge(fires_df, agency, left_on = "NWCG_REPORTING_UNIT_ID", right_on = "UnitId", how = "inner")
full_data
# fires_df.columns
# I droped containment_data as well as discovery time since most values were none
prediction_data = full_data[['STAT_CAUSE_CODE', 'FIRE_YEAR', 'FIRE_SIZE', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 
        'STATE', 'LATITUDE', 'LONGITUDE', 'OWNER_DESCR', 'GeographicArea' , 'Gacc','WildlandRole', 'UnitType', 'NWCG_REPORTING_UNIT_ID']]
# prediction_data.is()
# prediction_data.count()
# prediction_data
# Convert date_time Julian to Gregorian date
prediction_data['DISC_DATE'] = pd.to_datetime(prediction_data['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
prediction_data = prediction_data[['STAT_CAUSE_CODE', 'FIRE_YEAR', 'FIRE_SIZE', 'DISC_DATE', 'DISCOVERY_DOY', 
        'STATE', 'LATITUDE', 'LONGITUDE', 'OWNER_DESCR', 'GeographicArea' , 'Gacc','WildlandRole', 'UnitType', 'NWCG_REPORTING_UNIT_ID']]
# Extract day of the week and month
prediction_data['MONTH'] = pd.DatetimeIndex(prediction_data['DISC_DATE']).month
# 0 = Monday
prediction_data['DAY_OF_WEEK'] = prediction_data['DISC_DATE'].dt.dayofweek
# prediction_data
le = preprocessing.LabelEncoder()
# 'GeographicArea' , 'Gacc','WildlandRole', 'UnitType', 'NWCG_REPORTING_UNIT_ID',  ownder descr
prediction_data['STATE'] = le.fit_transform(prediction_data['STATE'])
prediction_data['OWNER_DESCR'] = le.fit_transform(prediction_data['OWNER_DESCR'])
prediction_data['GeographicArea'] = le.fit_transform(prediction_data['GeographicArea'])
prediction_data['Gacc'] = le.fit_transform(prediction_data['Gacc'])
prediction_data['WildlandRole'] = le.fit_transform(prediction_data['WildlandRole'])
prediction_data['UnitType'] = le.fit_transform(prediction_data['UnitType'])
prediction_data['NWCG_REPORTING_UNIT_ID'] = le.fit_transform(prediction_data['NWCG_REPORTING_UNIT_ID'])
prediction_data['OWNER_DESCR'] = le.fit_transform(prediction_data['OWNER_DESCR'])
correlation = prediction_data.corr()
fig = go.Figure(data = go.Heatmap(z = correlation, 
        x = correlation.columns,
        y = correlation.columns, 
        colorscale = 'RdBu'
                    ))
fig.update_layout(
    title = "Heatmap of chosen firedataset")
fig.show()
del fig
# Need to check for na values
# prediction_data.isna().sum()
# No na values which is expected since we did an inner join
# Using supervised learning to predict
prediction_data = prediction_data.drop('DISC_DATE', axis = 1)
prediction_data = prediction_data.dropna()
# prediction_data.head()
x = prediction_data.drop(['STAT_CAUSE_CODE'], axis = 1).values
y = prediction_data['STAT_CAUSE_CODE'].values
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 2)
dict_values = {1 : "Lightning", 2 : "Equipment Use", 3 : "Smoking", 4: "Campfire", 5 : "Debris Burning", 6: "Railroad"
              , 7: "Arson", 8 : "Children", 9 : "Miscellaneous", 10 : "Fireworks", 11 : "Powerline", 12 : "Structure", 
               13 : "Missing/Undefined"}
# Creating a baseline performance
# Notice: Imbalanced classes of labels

rfor = RandomForestClassifier(random_state=24, verbose=1, n_estimators = 110)
rfor = rfor.fit(x_train, y_train)
from collections import Counter
print(Counter([estimator.tree_.max_depth for estimator in rfor.estimators_]))
print(rfor.score(x_test, y_test))
# What features did it look at as important?
features = list(prediction_data.columns)[1:]
# print(features)
fig = go.Figure(
    data = go.Bar(x = features, y = rfor.feature_importances_, textposition='outside', 
        text = rfor.feature_importances_, texttemplate= "%{text: .4f}"),
    layout = go.Layout(
        title = 'Feature Importance',
        xaxis= dict(title = "Features", tickmode = 'linear'),
        yaxis = dict(title = "Importance", range = (0, 0.3))
    )   
)
fig.show()
del fig
### Using DART
# XGB BOOST AGGRESSIVELY Consumes memory when training a deep tree (taking from api doc)
# multi:softmax vs multi:softprob
# tree vs forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10,random_state = 2)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = 0.125, random_state = 3)
# x_train, y_train are for traiining, x_val,y_val are for validation, x_test, y_test are for testing
# This gives us a 80-10-10 split
xbst = XGBClassifier(
    learning_rate = 0.1, booster = 'gbtree', objective = 'multi:softmax',
    num_class = 13, random_state = 24, verbosity = 1, n_estimators = 3000
)
xbst.fit(x_train, y_train, early_stopping_rounds=3, eval_set = [(x_val,y_val)], eval_metric = 'mlogloss')
test_preds = xbst.predict(x_test)
predictions = [round(value) for value in test_preds]
print(accuracy_score(y_test, predictions))