# Importing relevant libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb
import matplotlib.dates as mdates

import plotly.express as px

from datetime import date, timedelta

from sklearn.cluster import KMeans

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm

from keras.models import Sequential

from keras.layers import LSTM,Dense

from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from statsmodels.tsa.seasonal import seasonal_decompose
# Importing the datasets into a pandas dataframe

patient=pd.read_csv('../input/coronavirusdataset/patient.csv')

time=pd.read_csv('../input/coronavirusdataset/time.csv')

route=pd.read_csv('../input/coronavirusdataset/route.csv')

trend=pd.read_csv('../input/coronavirusdataset/trend.csv')

case=pd.read_csv('../input/coronavirusdataset/case.csv')
patient.info()
# Descriptive statistics of the patients table

patient.describe()
# First few rows of the table

patient.head()
# Checking for nulls in different columns

patient.isna().sum()
# Creating a copy of patient table

df=patient.copy()
# Changing the data type of year to int type

patient['birth_year'] = patient.birth_year.fillna(0.0).astype(int)

patient['birth_year'] = patient['birth_year'].map(lambda val: val if val > 0 else np.nan)
patient['age']=2020-patient['birth_year']
import math

def age_group(age):

    if(age>=0):

        if age % 10 != 0:

            lower=(math.floor(age/10))*10

            upper=(math.ceil(age/10))*10-1

            return(str(lower)+'-'+str(upper))

        else:

            lower = int(age)

            upper = int(age + 9) 

            return f"{lower}-{upper}"

    return('Unknown')
patient["age_group"] = patient["age"].apply(age_group)
patient['confirmed_date']=pd.to_datetime(patient['confirmed_date'])

patient['released_date']=pd.to_datetime(patient['released_date'])

patient['deceased_date']=pd.to_datetime(patient['deceased_date'])
patient['release_time']=patient['released_date']-patient['confirmed_date']

patient['death_time']=patient['deceased_date']-patient['confirmed_date']

patient["duration_since_confirmed"] = patient[["release_time", "death_time"]].min(axis=1)

patient["duration_days"] = patient["duration_since_confirmed"].dt.days
# Calculating the percentage of recovery,isolated and deceased patients.

rp = patient.loc[patient["state"] == "released"].shape[0]

dp = patient.loc[patient["state"] == "deceased"].shape[0]

ip = patient.loc[patient["state"]== "isolated"].shape[0]

rp=rp/patient.shape[0]

dp=dp/patient.shape[0]

ip=ip/patient.shape[0]

print("The percentage of recovery is "+ str(rp*100) )

print("The percentage of deceased is "+ str(dp*100) )

print("The percentage of isolated is "+ str(ip*100) )
k=patient['state'].value_counts()

k.plot(kind='pie',figsize=(20,10),legend=True)

plt.legend(loc=0,bbox_to_anchor=(1.5,0.5));
k=patient['infection_reason'].value_counts()

k.plot(kind='pie',figsize=(20,10),legend=True)

plt.legend(loc=0,bbox_to_anchor=(2.0,0.5));
# Splitting the data into three parts depending upo their state.

released=patient[patient['state']=='released']

deceased=patient[patient['state']=='deceased']

isolated=patient[patient['state']=='isolated']
# Creating a new column of state by gender

patient["state_by_gender"] = patient["state"] + "_" + patient["sex"]
plt.figure(figsize=[15,10])

sb.barplot(x='country',y='age',hue='state',data=patient)

plt.legend(loc='best');
plt.figure(figsize=[15,10])

sb.barplot(x='country',y='age',hue='sex',data=deceased)

plt.legend(loc='best');
plt.figure(figsize=[12,7])

sb.countplot(data=patient,x='state',hue='sex');
plt.figure(figsize=[12,7])

sb.countplot(data=patient,x='state',hue='infection_reason');
plt.figure(figsize=[12,7])

sb.violinplot(data=patient,x='state',y='age',hue='sex')

plt.show();
plt.figure(figsize=[12,7])

sb.violinplot(data=patient,x='sex',y='age')

plt.show();
plt.figure(figsize=[12,7])

sb.violinplot(data=patient,x='country',y='age',hue='sex')

plt.show();
daily_count = patient.groupby(patient.confirmed_date).patient_id.count()
accumulated_count = daily_count.cumsum()
plt.figure(figsize=[25,7])

accumulated_count.plot()

plt.title('Accumulated Confirmed Count');
sb.kdeplot(data=deceased['age'],label='deceased', shade=True)

sb.kdeplot(data=released['age'],label='released', shade=True)

sb.kdeplot(data=isolated['age'],label='isolated', shade=True);
plt.figure(figsize=(15, 5))

plt.title('Sex')

deceased.sex.value_counts().plot.bar();
plt.figure(figsize=[18,7])

sb.countplot(data=patient,x='age_group',hue='state_by_gender')

plt.legend(loc='best');
sb.set_style("whitegrid")

sb.FacetGrid(patient, hue = 'state', height = 10).map(plt.scatter, 'age', 'region').add_legend()

plt.title('Region by age and state')

plt.show()
plt.figure(figsize=[12,7])

sb.countplot(data=deceased,x='sex',hue='disease');
plt.figure(figsize=[12,7])

sb.countplot(data=patient,x='state',hue='disease');
route.head()
route.info()
clus=route.loc[:,['id','latitude','longitude']]

clus.head(10)
K_clusters = range(1,8)

kmeans = [KMeans(n_clusters=i) for i in K_clusters]

Y_axis = route[['latitude']]

X_axis = route[['longitude']]

score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

plt.plot(K_clusters, score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.show()
kmeans = KMeans(n_clusters = 4, init ='k-means++')

kmeans.fit(clus[clus.columns[1:3]])

clus['cluster_label'] = kmeans.fit_predict(clus[clus.columns[1:3]])

centers = kmeans.cluster_centers_

labels = kmeans.predict(clus[clus.columns[1:3]])
clus.plot.scatter(x = 'latitude', y = 'longitude', c=labels, s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5);
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon,city in zip(route['latitude'], route['longitude'],route['city']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
plt.figure(figsize=(15,5))

plt.title('Number patients in city')

route.city.value_counts().plot.bar();
plt.figure(figsize=(15,5))

plt.title('Number patients in Province/State')

route.province.value_counts().plot.bar();
plt.figure(figsize=(15,5))

plt.title('Visit')

route.visit.value_counts().plot.bar();
plt.figure(figsize=(12, 8))

sb.boxplot(x="sex",

            y="duration_days",hue='state',

            hue_order=["released", "deceased"],

            data=patient)

plt.title("Time from confirmation to release or death");
plt.figure(figsize=(12, 8))

sb.boxplot(x="age_group",

            y="duration_days",hue='state',

            hue_order=["released", "deceased"],

            data=patient)

plt.title("Time from confirmation to release or death");
trend.describe()
trend_cold=trend[['date','cold']]

trend_flu=trend[['date','flu']]

trend_pneumonia=trend[['date','pneumonia']]

trend_coronavirus=trend[['date','coronavirus']]
trend_cold['date']=pd.to_datetime(trend_cold['date'])

trend_cold.index=trend_cold['date']

trend_cold.drop(['date'],axis=1,inplace=True)

trend_flu['date']=pd.to_datetime(trend_flu['date'])

trend_flu.index=trend_flu['date']

trend_flu.drop(['date'],axis=1,inplace=True)

trend_pneumonia['date']=pd.to_datetime(trend_pneumonia['date'])

trend_pneumonia.index=trend_pneumonia['date']

trend_pneumonia.drop(['date'],axis=1,inplace=True)

trend_coronavirus['date']=pd.to_datetime(trend_coronavirus['date'])

trend_coronavirus.index=trend_coronavirus['date']

trend_coronavirus.drop(['date'],axis=1,inplace=True)
decomposition = seasonal_decompose(trend_cold) 

trend_cld = decomposition.trend

plt.figure(figsize=(18, 8))

plt.plot(trend_cld, label='Trend')

plt.title('Trend of Cold')

plt.legend(loc='best');
plt.figure(figsize=(18, 8))

decomposition = seasonal_decompose(trend_flu) 

trend_fl = decomposition.trend

plt.plot(trend_fl, label='Trend')

plt.title('Trend of flu')

plt.legend(loc='best');
plt.figure(figsize=(18, 8))

decomposition = seasonal_decompose(trend_pneumonia) 

trend_pneu = decomposition.trend

plt.plot(trend_pneu, label='Trend')

plt.title('Trend of Pneumonia')

plt.legend(loc='best');
plt.figure(figsize=(18, 8))

decomposition = seasonal_decompose(trend_coronavirus) 

trend_corona = decomposition.trend

plt.plot(trend_corona, label='Trend')

plt.title('Trend of Coronavirus')

plt.legend(loc='best');