import pandas as pd # for data analyse and data manupulation

import matplotlib.pyplot as plt # visualization

import numpy as np  

import folium # visualization

import seaborn as sns # visualization
data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

data.head()

data.describe()

groups = data.groupby(['grade'])['price'].mean()

plt.figure(figsize=(10, 5))

plt.xlabel('price')

groups.plot.barh()

groups = data.groupby(['bedrooms'])['price'].mean()

plt.figure(figsize=(10, 5))

plt.xlabel('price')

groups.plot.barh()
groups = data.groupby(['bathrooms'])['price'].mean()

plt.figure(figsize=(10, 10))

groups.plot.barh()
sns.countplot(data.bathrooms, order = data['bathrooms'].value_counts().index)

sns.countplot(data.bedrooms, order = data['bedrooms'].value_counts().index)

sns.countplot(data.grade, order = data['grade'].value_counts().index)

sns.countplot(data.condition, order = data['condition'].value_counts().index)

def generateBaseMap(map_location=[47.5,-122.161], zoom=9):

    base_map = folium.Map(location=map_location, control_scale=True, zoom_start=zoom)

    return base_map

from folium.plugins import HeatMap

df_copy = data[np.logical_and(data.yr_built<=1980,data.yr_built >= 1970)] 

df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)

base_map

df_copy = data[np.logical_and(data.yr_built<=1990,data.yr_built >= 1980)] 

df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)

base_map
df_copy = data[np.logical_and(data.yr_built<=2000,data.yr_built >= 1990)] 

df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)

base_map
df_copy = data[np.logical_and(data.yr_built<=2010,data.yr_built >= 2000)] 

df_copy['count'] = 1

base_map = generateBaseMap()

HeatMap(data=df_copy[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)

base_map
neededCols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',

            'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',

            'yr_renovated', 'sqft_living15', 'sqft_lot15']





corr = data[neededCols].corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
dataForRegression = data[neededCols]
dataForRegression.head()
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split



import statsmodels.api as sm
X=dataForRegression.drop('price',axis=1)

y=dataForRegression['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=42)

lm = linear_model.LinearRegression() 

model = lm.fit(X_train[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',

       'waterfront', 'view', 'condition', 'grade', 'sqft_above',

       'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15',

       'sqft_lot15']], y_train)



lm = sm.OLS(y_train, X_train)

model1 = lm.fit()

model1.summary()




print('model accuracy is : ',model.score(X_test,y_test))
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.metrics import mean_squared_error, r2_score

np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
cross_val_score(model, X_train, y_train, cv = 100, scoring = "r2").mean()



predictedDatas=[]

for  row in range(0,len(dataForRegression)):

    a=(model.predict([[dataForRegression['bedrooms'].values[row],dataForRegression['bathrooms'].values[row],dataForRegression['sqft_living'].values[row],dataForRegression['sqft_lot'].values[row],dataForRegression['floors'].values[row],

        dataForRegression['waterfront'].values[row],dataForRegression['view'].values[row],dataForRegression['condition'].values[row],dataForRegression['grade'].values[row],dataForRegression['sqft_above'].values[row],

        dataForRegression['sqft_basement'].values[row],dataForRegression['yr_built'].values[row],dataForRegression['yr_renovated'].values[row],dataForRegression['sqft_living15'].values[row],dataForRegression['sqft_lot15'].values[row]

        ]]))

    a=round(a[0],0)

    predictedDatas.append(a)



final_df = dataForRegression.price.values

final_df = pd.DataFrame(final_df,columns=['Real_price'])

final_df['predicted_prices'] = predictedDatas

final_df['difference'] = abs(final_df['Real_price'] - final_df['predicted_prices'])

final_df.tail(20)
prediction= model.predict([[2,0,1180,6000,1,0,0,4,7,1180,0,1995,2010,1340,6000]]) 

prediction=round(prediction [0],0)

prediction