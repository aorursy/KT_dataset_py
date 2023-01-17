# Importar las librerias a utilizar 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Importamos la libreria folium que nos permite trabajar con mapas

import folium

from folium import plugins

from folium.plugins import HeatMap



# Importamos los modelos a utilizar

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from xgboost import XGBRegressor



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



import plotly.graph_objects as go

import plotly.express as px



# La siguiente línea es para ver las imagenes dentro del notebook

%matplotlib inline



# Acá configuramos el tamaño de las figuras

plt.rcParams['figure.figsize'] = (12,8)



# Seteamos opciones de pandas sobre las columnas y su ancho

pd.set_option('max_columns', 120)

pd.set_option('max_colwidth', 5000)



import warnings

warnings.filterwarnings('ignore')

# Cargamos el dataset de Airbnb de New York

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head(5)
data.shape
data.dtypes
data.drop(['host_name','name'], axis=1, inplace=True)
data.isnull().sum() 
data[data['number_of_reviews']== 0.0].shape
data['last_review'].fillna(0, inplace=True)

data['reviews_per_month'].fillna(0, inplace=True)
data['last_review'] = pd.to_datetime(data['last_review'],infer_datetime_format=True) 
data['all_year_avail'] = data['availability_365']>353

data['low_avail'] = data['availability_365']< 12

data['no_reviews'] = data['reviews_per_month']==0
data.dtypes
percentdist = data['neighbourhood_group'].value_counts()



neighbourhood_group = pd.DataFrame({"col_label": percentdist.index, "col_values": percentdist.values})

fig = px.pie(neighbourhood_group, values='col_values', names='col_label', color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.availability_365,palette=cmap)

plt.show()
data['all_year_avail'] = data['availability_365']>353

data['low_avail'] = data['availability_365']< 12

data['no_reviews'] = data['reviews_per_month']==0
percentroom = data['room_type'].value_counts()



roomtype = pd.DataFrame({"room_label": percentroom.index, "room_values": percentroom.values})

fig = px.pie(roomtype, values='room_values', names='room_label', color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.room_type)

plt.show()
mean_price=data.price.mean()

print("Precio Promedio =", mean_price)
pd.DataFrame(data.groupby('neighbourhood_group')["price"].mean().sort_values(ascending=False))
map_newyork = folium.Map(location=[40.6643, -73.9385], zoom_start = 10, control_scale = True)

data['latitude'] = data['latitude'] .astype(float)

data['longitude'] = data['longitude'].astype(float)

heat_df = data.dropna(axis=0, subset=['latitude','longitude'])

heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

def generateBaseMap(default_location=[40.6643, -73.9385], default_zoom_start=10):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map



base_map = generateBaseMap()

HeatMap(data=data[['latitude', 'longitude', 'price']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)

base_map
fig = px.scatter(data, x="neighbourhood_group", y="price", color="neighbourhood_group")

fig.show()
df_manhattan = data[data['neighbourhood_group']=='Manhattan']
fig = px.scatter(df_manhattan, x="neighbourhood", y="price", color="neighbourhood")

fig.show()
pd.DataFrame(df_manhattan.groupby('neighbourhood')["price"].mean().sort_values(ascending=False))
def outliers(dataout):

    median=dataout.median()

    p25 = dataout.quantile(0.25)

    p75 = dataout.quantile(0.75)

    iqr = p75 - p25

    li= p25 -1.5*iqr 

    ls= p75 +1.5*iqr

    lie=p25 -3*iqr 

    lse=p75 +3*iqr

    print("Rango Intercuartilico =",iqr) #Rango Intercuartilico

    print("Limite Inferior =",li)

    print("Limite Superior =",ls)

    print("Limite Inferior Extremo =",lie)

    print("Limite Superior Extremo =",lse)

dataout=data['price']

outliers(dataout)
outliers_price=data[data['price']>334]
outliers = folium.map.FeatureGroup()



for lat, lng, in zip(outliers_price.latitude, outliers_price.longitude):

    outliers.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=4, 

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.4

        )

    )



map_newyork.add_child(outliers)
model_data = data[(data['price']) < 334]
corr = model_data.corr(method='pearson')

plt.figure(figsize=(15,8))

sns.heatmap(corr, annot=True)
dataset = pd.get_dummies(model_data,columns= ['neighbourhood_group','neighbourhood', 'room_type'])
X = dataset.drop(['price','id', 'host_id', 'last_review', 'last_review'], axis=1)

y = dataset['price']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=42)



# Realizá la separación a continuación en esta celda
num_folds = 10

scoring = "neg_mean_squared_error"



models = []

models.append(('XGBR', XGBRegressor(objective="reg:squarederror")))

models.append(('KNN', KNeighborsRegressor()))

models.append(('DTR', DecisionTreeRegressor()))

models.append(('RFR', RandomForestRegressor()))

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds,shuffle=True, random_state=42)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, np.sqrt(-cv_results.mean()),   cv_results.std())

    print(msg)
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)



params = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }





model = XGBRegressor(objective="reg:squarederror")

kfold = KFold(n_splits=5, random_state=42)

grid=RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=-1, scoring=scoring, cv=kfold, 

                        return_train_score=True)



grid_result = grid.fit(rescaledX, y_train)





mean_train = np.sqrt(-grid_result.cv_results_['mean_train_score'].mean())

mean_test = np.sqrt(-grid_result.cv_results_['mean_test_score'].mean())



print("Best Params: " ,grid_result.best_params_)

print("Mean Train Score: ", mean_train)

print("Mean Test Score: ", mean_test)
rescaledXtest = scaler.transform(X_test)

optimized=grid_result.best_estimator_

ypredtrain = optimized.predict(rescaledX)

ypredtest = optimized.predict(rescaledXtest)

r2_train = r2_score(y_train, ypredtrain)

r2_test = r2_score(y_test, ypredtest)



print("r2 Train :",r2_train)

print("r2 Test :",r2_test)