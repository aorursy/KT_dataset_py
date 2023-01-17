import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap

import datetime

from datetime import date

import math

from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects

from sklearn.metrics import mean_squared_error

from sklearn.cluster import SpectralClustering

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn import preprocessing

from random import sample 

import plotly.express as px



sns.set(rc={'figure.figsize':(12,10)})

sns.set(style="white", context="talk")



%matplotlib inline
arrest = pd.read_csv("../input/nypd-crime/NYPD_Arrest_Data__Year_to_Date_ (1).csv")

arrest[:5]
def date_to_weekday(date):

    weekday_dict = {0:'Monday', 1:'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    date_time_obj = datetime.datetime.strptime(date, '%m/%d/%Y')

    return weekday_dict[date_time_obj.weekday()]

def code_to_loc(code):

    code_dict = {'B': 'Bronx', 'S': 'Staten Island', 'K': 'Brooklyn', 'M': 'Manhattan' , 'Q': 'Queens'}

    return code_dict[code]

def code_to_fel(code):

    code_dict = {'F': 'Felony','M': 'Misdemeanor', 'V': 'Violation', 'I': 'Other'}

    if code in code_dict:

        return code_dict[code]

    else:

        return 'Other'



date = arrest['ARREST_DATE'].str.split("/", n = 3, expand = True)

arrest['year'] = date[2].astype('int32')

arrest['day'] = date[1].astype('int32')

arrest['month'] = date[0].astype('int32')



arrest['ARREST_BORO'] = arrest['ARREST_BORO'].apply(code_to_loc)

arrest['WEEKDAY'] = arrest['ARREST_DATE'].apply(date_to_weekday)

arrest['LAW_CAT_CD'] = arrest['LAW_CAT_CD'].apply(code_to_fel)



arrest = arrest.drop(['ARREST_KEY', 'PD_CD', 'PD_DESC', 'KY_CD', 'LAW_CODE', 'JURISDICTION_CODE', 'X_COORD_CD', 'Y_COORD_CD'], axis=1)

arrest[:5]
f, ax = plt.subplots(figsize=(25, 15))

sns.countplot(y="day", data=arrest, palette="pastel");
f, ax = plt.subplots(figsize=(10, 7))

sns.countplot(x="month", data=arrest)
sns.catplot(x="PERP_RACE", data=arrest,kind="count", palette="pastel", height=15, aspect=1.5);
f, ax = plt.subplots(figsize=(10, 8))

sns.countplot(y="AGE_GROUP", data=arrest, palette="pastel");
## Borough of arrest. B(Bronx), S(Staten Island), K(Brooklyn), M(Manhattan), Q(Queens)

f, ax = plt.subplots(figsize=(10, 8))

sns.countplot(x="ARREST_BORO", data=arrest, palette="pastel");
ax = sns.catplot(x="PERP_RACE", hue="PERP_SEX", kind="count",palette="cubehelix", data=arrest, height=15, aspect=2)
sns.catplot(x="ARREST_BORO", kind="count",hue="PERP_SEX",palette="Set2", data=arrest,height=12, aspect = 2);
ax = sns.catplot(x="AGE_GROUP", hue="PERP_SEX", kind="count",palette="cubehelix", data=arrest, height=10, aspect = 2)
ax = sns.catplot(x="ARREST_BORO", hue="PERP_RACE", kind="count",palette="cubehelix", data=arrest, height=10, aspect = 2)
ax = sns.catplot(x="WEEKDAY", hue="ARREST_BORO", kind="count",palette="bright", data=arrest, height=10, aspect = 2)

ax = sns.catplot(x="WEEKDAY", hue="AGE_GROUP", kind="count",palette="rainbow", data=arrest, height=10, aspect = 2)
# positions = [] 

# for index, row in arrest.iterrows():

positions = list(zip(arrest['Latitude'], arrest['Longitude']))

tiles = 'Stamen Terrain'

fol = folium.Map(location=[40.75,-73.98], zoom_start=10, tiles = tiles)

pos_samp = sample(positions, 22000)#22K is the max now as we join both DS togather 

HeatMap(pos_samp, radius = 8).add_to(fol) 

fol
px.set_mapbox_access_token(open("../input/map-key/key").read()) ## key needed for the API of the maps in plotly

arrest["size"] = 1

fig = px.scatter_mapbox(arrest, lat="Latitude", lon="Longitude", color="AGE_GROUP", size="size", animation_frame="month",

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)

fig.show()

arrest = arrest.drop('size', axis = 1)
edu = pd.read_csv("../input/nypd-crime/DYCD_after-school_programs (1).csv")

edu[:5]
boro_list = ['Bronx', 'Staten Island', 'Brooklyn', 'Manhattan' , 'Queens']

edu_boro = edu[edu['BOROUGH / COMMUNITY'].isin(boro_list)]
names = edu_boro.groupby('BOROUGH / COMMUNITY').count().index

my_circle=plt.Circle( (0,0), 0.7, color='white')

f, ax = plt.subplots(figsize=(15, 12))

cmap = plt.get_cmap('Spectral')

colors = [cmap(i) for i in np.linspace(0, 1, 6)]

plt.pie(edu_boro.groupby('BOROUGH / COMMUNITY').count()['Postcode'], labels=names, colors=colors, shadow=True)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
positions = [] 

for index, row in edu.iterrows():

    if not math.isnan(row['Latitude']) :

        positions.append((row['Latitude'], row['Longitude']))

fol = folium.Map(location=[40.75,-73.98],tiles='Stamen Toner', zoom_start=11)

HeatMap(positions[:38000], radius = 8).add_to(fol)

fol
positions_edu = [] 



for index, row in edu.iterrows():

    if not math.isnan(row['Latitude']) :

        positions_edu.append((row['Latitude'], row['Longitude']))

positions_arr = list(zip(arrest['Latitude'], arrest['Longitude']))

fol = folium.Map(location=[40.75,-73.98], zoom_start=11, control_scale=True)



pos_samp = sample(positions_arr, 22000)#22K is the max now as we join both DS togather 

HeatMap(pos_samp, radius = 7).add_to(fol) 



for pos in positions_edu:

    folium.CircleMarker(location=[pos[0],pos[1]], radius=1, color='red', fill=False,).add_to(fol)

fol
def cat_to_num(df , col_name):

    le = preprocessing.LabelEncoder()

    new_col = le.fit_transform(df[col_name])

    return le , new_col



est1 = RandomForestRegressor()

model1 = MultiOutputRegressor(est1, 4)



est2 = GradientBoostingRegressor()

model2 = MultiOutputRegressor(est2, 4)



x_reg_train = arrest[arrest['month']<6].drop(['LAW_CAT_CD','ARREST_DATE','OFNS_DESC','ARREST_PRECINCT', 'day','year'],axis = 1)

x_reg_test = arrest[arrest['month']==6].drop(['LAW_CAT_CD','ARREST_DATE','OFNS_DESC','ARREST_PRECINCT', 'day', 'year'],axis = 1)



y_train = x_reg_train[['Latitude', 'Longitude']]

y_test = x_reg_test[['Latitude', 'Longitude']]



x_reg_train = x_reg_train.drop(['Latitude', 'Longitude'],axis = 1)

x_reg_test = x_reg_test.drop(['Latitude', 'Longitude'],axis = 1)



d = {}

for col in x_reg_train.columns:

    if x_reg_train.dtypes[col] == 'int32':

        continue

    le, new_col = cat_to_num(x_reg_train, col)

    d[col] = le

    x_reg_train[col] = new_col

    x_reg_test[col] = le.transform(x_reg_test[col])

model1.fit(x_reg_train, y_train)

model2.fit(x_reg_train, y_train)
pred1 = model1.predict(x_reg_test)

pred2 = model2.predict(x_reg_test)

y_test_mean = y_test.mean(axis=0).tolist()

print("The MSE of RF = {}\nThe MSE of GBR = {} ".format(mean_squared_error(y_test, pred1),mean_squared_error(y_test, pred2)))
future_crime = pd.DataFrame({

    'ARREST_BORO':['Brooklyn','Brooklyn', 'Manhattan','Queens', 'Bronx'], 

    'PERP_SEX':['M', 'F' , 'F', 'M' , 'F'], 

    'PERP_RACE':['BLACK', 'WHITE', 'WHITE HISPANIC', "WHITE" , 'ASIAN / PACIFIC ISLANDER'], 

    'WEEKDAY':['Sunday', 'Monday', 'Tuesday', 'Sunday', 'Friday'], 

    'AGE_GROUP':['<18', '25-44', '18-24', '45-64', '25-44'],

    'month':[1,3,4,5,2]

    })

for col in future_crime.columns:

    if future_crime.dtypes[col] == 'int64':

        continue

    future_crime[col] = d[col].transform(future_crime[col])

cords1 = model1.predict(future_crime)

cords2 = model2.predict(future_crime)
positions_arr= list(zip(arrest['Latitude'], arrest['Longitude']))



fol = folium.Map(location=[40.75,-73.98], zoom_start=11, control_scale=True)



pos_samp = sample(positions_arr, 22000)#22K is the max now as we join both DS togather 

HeatMap(pos_samp, radius = 9).add_to(fol) 



for pos in cords1:

    folium.CircleMarker(location=[pos[0],pos[1]], radius=3, color='red', fill=True).add_to(fol)

for pos in cords2:

    folium.CircleMarker(location=[pos[0],pos[1]], radius=3, color='blue', fill=True).add_to(fol)

fol