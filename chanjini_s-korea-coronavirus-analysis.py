import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

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
path = '/kaggle/input/coronavirusdataset/'



case = pd.read_csv(path+'Case.csv')

p_info = pd.read_csv(path+'PatientInfo.csv')

p_route = pd.read_csv(path+'PatientRoute.csv')

time = pd.read_csv(path+'Time.csv')

t_age = pd.read_csv(path+'TimeAge.csv')

t_gender = pd.read_csv(path+'TimeGender.csv')

t_provin = pd.read_csv(path+'TimeProvince.csv')

region = pd.read_csv(path+'Region.csv')

weather = pd.read_csv(path+'Weather.csv')

search = pd.read_csv(path+'SearchTrend.csv')
case.head()
caseList = case['infection_case'].unique()

columns = ['total_confirmed']

caseTotal = pd.DataFrame(index = caseList, columns = columns)

for i in range(len(caseList)):

    caseTotal.loc[caseList[i]] = case[case['infection_case'] == caseList[i]]['confirmed'].sum()

caseTotal = caseTotal.sort_values(by=['total_confirmed'], ascending=False)

caseTotal
p_info.head()
case[case['infection_case'] == 'overseas inflow']['confirmed'].sum()
p_info['infection_case'].value_counts()
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=p_info['state'].loc[

    (p_info['infection_case']=='overseas inflow')

])
p_info['country'].value_counts()
inflow_p_info = p_info[p_info['infection_case'] == 'overseas inflow']
inflow_p_info['country'].value_counts()
china_inflow = p_info[p_info['country'] == 'China']

china_inflow = china_inflow[china_inflow['infection_case'] == 'overseas inflow']

china_inflow = china_inflow.reset_index(drop=True)

china_inflow
infectedby = p_info[p_info['infected_by'].notna()]

infectedby = infectedby.reset_index(drop=True)

infectedby.shape
infectedbyList = infectedby['infected_by'].isin(china_inflow['patient_id']).replace({False:np.nan}).dropna().index
infectedbyChinese = pd.DataFrame(infectedby, index=infectedbyList)

infectedbyChinese
infectedbyList2 = infectedby['infected_by'].isin(infectedbyChinese['patient_id']).replace({False:np.nan}).dropna().index
infectedbyChinese2 = pd.DataFrame(infectedby, index=infectedbyList2)

infectedbyChinese2
US_inflow = p_info[p_info['country'] == 'United States']

US_inflow
infectedbyList = infectedby['infected_by'].isin(US_inflow['patient_id']).replace({False:np.nan}).dropna().index

infectedbyUS = pd.DataFrame(infectedby, index=infectedbyList)

infectedbyUS
france_inflow = p_info[p_info['country'] == 'France']

france_inflow
infectedbyList = infectedby['infected_by'].isin(france_inflow['patient_id']).replace({False:np.nan}).dropna().index

infectedbyFrance = pd.DataFrame(infectedby, index=infectedbyList)

infectedbyFrance
thailand_inflow = p_info[p_info['country'] == 'Thailand']

thailand_inflow
infectedbyList = infectedby['infected_by'].isin(thailand_inflow['patient_id']).replace({False:np.nan}).dropna().index

infectedbyThailand = pd.DataFrame(infectedby, index=infectedbyList)

infectedbyThailand
switz_inflow = p_info[p_info['country'] == 'Switzerland']

switz_inflow
infectedbyList = infectedby['infected_by'].isin(switz_inflow['patient_id']).replace({False:np.nan}).dropna().index

infectedbySwitz = pd.DataFrame(infectedby, index=infectedbyList)

infectedbySwitz
mongolia_inflow = p_info[p_info['country'] == 'Mongolia']

mongolia_inflow
infectedbyList = infectedby['infected_by'].isin(mongolia_inflow['patient_id']).replace({False:np.nan}).dropna().index

infectedbyMongolia = pd.DataFrame(infectedby, index=infectedbyList)

infectedbyMongolia
koreanPatient = p_info[p_info['country'] == 'Korea']
koreanPatient['infection_case'].value_counts()
foreignVisit = ['overseas inflow', 'Pilgrimage to Israel']
korea_inflow = koreanPatient.loc[koreanPatient['infection_case'].isin(foreignVisit)]
korea_inflow.shape[0]
korea_inflow.head()
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=korea_inflow['state'])
plt.figure(figsize=(13, 8))

plt.title('Korean overseas inflow patients province')

korea_inflow.province.value_counts(ascending=True).plot.barh()

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
plt.figure(figsize=(13, 8))

plt.title('Korean overseas inflow patients age')

korea_inflow.age.value_counts(ascending=True).plot.barh()

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=korea_inflow['sex'])
infectedbyList = infectedby['infected_by'].isin(korea_inflow['patient_id']).replace({False:np.nan}).dropna().index

infectedbyKorean = pd.DataFrame(infectedby, index=infectedbyList)

infectedbyKorean.shape[0]
infectedbyKorean.head()
infectedbyList2 = infectedby['infected_by'].isin(infectedbyKorean['patient_id']).replace({False:np.nan}).dropna().index

infectedbyKorean2 = pd.DataFrame(infectedby, index=infectedbyList2)

infectedbyKorean2
infectedbyList3 = infectedby['infected_by'].isin(infectedbyKorean2['patient_id']).replace({False:np.nan}).dropna().index

infectedbyKorean3 = pd.DataFrame(infectedby, index=infectedbyList3)

infectedbyKorean3
case.head()
provinceList = case['province'].unique()

columns = ['total_confirmed']

provinceTotal = pd.DataFrame(index = provinceList, columns = columns)

for i in range(len(provinceList)):

    provinceTotal.loc[provinceList[i]] = case[case['province'] == provinceList[i]]['confirmed'].sum()

provinceTotal = provinceTotal.sort_values(by=['total_confirmed'], ascending=True)
provinceTotal.tail()
dataFrame = pd.DataFrame(data=provinceTotal, index=provinceTotal.index);

dataFrame.plot.barh(figsize=(20,10));

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)
cityList = case['city'].unique()

columns = ['total_confirmed']

cityTotal = pd.DataFrame(index = cityList, columns = columns)

for i in range(len(cityList)):

    cityTotal.loc[cityList[i]] = case[case['city'] == cityList[i]]['confirmed'].sum()

cityTotal = cityTotal.sort_values(by=['total_confirmed'], ascending=True)
dataFrame = pd.DataFrame(data=cityTotal, index=cityTotal.index);

dataFrame.plot.barh(figsize=(20,10));

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)
caseList = case['infection_case'].unique()

columns = ['total_confirmed']

caseTotal = pd.DataFrame(index = caseList, columns = columns)

for i in range(len(caseList)):

    caseTotal.loc[caseList[i]] = case[case['infection_case'] == caseList[i]]['confirmed'].sum()

caseTotal = caseTotal.sort_values(by=['total_confirmed'], ascending=True)
caseTotal.tail()
dataFrame = pd.DataFrame(data=caseTotal, index=caseTotal.index);

dataFrame.plot.barh(figsize=(15,18));

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=p_info['state'].loc[

    (p_info['infection_case']=='Shincheonji Church')

])
sns.set(rc={'figure.figsize':(5,5)})

sns.countplot(x=p_info['state'].loc[

    (p_info['infection_case']=='etc')

])
columns = ['group']

caseGroup = pd.DataFrame(index = caseList, columns = columns)

for i in range(len(caseList)):

    caseGroup.loc[caseList[i]] = case[case['infection_case'] == caseList[i]]['group'].values[0]
caseAnalysis = pd.concat([caseTotal, caseGroup], axis=1, sort=False)

caseAnalysis.sort_values(by=['total_confirmed'], ascending=False)
caseGroup = caseAnalysis[caseAnalysis['group'] == True]['total_confirmed'].sum()
caseNotGroup = caseAnalysis[caseAnalysis['group'] == False]['total_confirmed'].sum()
index = ['group', 'not group']

column = ['total_confirmed']

df = pd.DataFrame(index=index, columns=column)

df.loc['group']['total_confirmed'] = caseGroup

df.loc['not group']['total_confirmed'] = caseNotGroup

df
plot = df.plot.pie(y='total_confirmed', figsize=(5, 5))

plt.title("Group vs. Non-group")
cityTotal['city'] = cityTotal.index

cityTotal = cityTotal.reset_index(drop=True)

cityTotal = cityTotal.sort_values(by=['total_confirmed'], ascending=False)
caseTemp = case.loc[:, ['city', 'latitude', 'longitude']]

clus = cityTotal.merge(caseTemp, on='city')

clus = clus.sort_values(by=['total_confirmed'], ascending=False)

clus = clus.drop(clus[clus['latitude'] == '-'].index)

clus = clus.drop_duplicates('city')

clus = clus.reset_index(drop=True)
clus
clus['longitude'] = pd.to_numeric(clus['longitude'])

clus['latitude'] = pd.to_numeric(clus['latitude'])
from shapely.geometry import Point



geometry = [Point(xy) for xy in zip(clus['longitude'], clus['latitude'])]

geometry[1:3]
import geopandas as gpd

crs = {'init': 'epsg:4326'}

geo_df = gpd.GeoDataFrame(clus, crs=crs, geometry=geometry)

geo_df
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon, city, total in zip(geo_df['latitude'], geo_df['longitude'], geo_df['city'], geo_df['total_confirmed']):

    folium.CircleMarker([lat, lon],

                        radius=int(total/100),

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
p_route.head()
chinaInflowList = p_route['patient_id'].isin(china_inflow['patient_id']).replace({False:np.nan}).dropna().index
koreaInflowList = p_route['patient_id'].isin(korea_inflow['patient_id']).replace({False:np.nan}).dropna().index
china_inflow_route = pd.DataFrame(p_route, index=chinaInflowList)

china_inflow_route
korea_inflow_route = pd.DataFrame(p_route, index=koreaInflowList)

korea_inflow_route.head()
p_info['infection_case'].unique()
SCJChurchList = p_route['patient_id'].isin(p_info[p_info['infection_case'] == 'Shincheonji Church']['patient_id']).replace({False:np.nan}).dropna().index

SCJChurch_route = pd.DataFrame(p_route, index=SCJChurchList)

SCJChurch_route.shape
contactPList = p_route['patient_id'].isin(p_info[p_info['infection_case'] == 'contact with patient']['patient_id']).replace({False:np.nan}).dropna().index

contactP_route = pd.DataFrame(p_route, index=contactPList)

contactP_route.shape
clus=p_route.loc[:,['patient_id','latitude','longitude']]

clus.head(10)
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon,city in zip(p_route['latitude'], p_route['longitude'], p_route['city']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
clus=china_inflow_route.loc[:,['patient_id','latitude','longitude']]
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon,city in zip(china_inflow_route['latitude'], china_inflow_route['longitude'], china_inflow_route['city']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
clus=korea_inflow_route.loc[:,['patient_id','latitude','longitude']]
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon,city in zip(korea_inflow_route['latitude'], korea_inflow_route['longitude'], korea_inflow_route['city']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
clus=SCJChurch_route.loc[:,['patient_id','latitude','longitude']]
SCJChurch_route['patient_id'].unique()
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon,city in zip(SCJChurch_route['latitude'], SCJChurch_route['longitude'], SCJChurch_route['city']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
clus=contactP_route.loc[:,['patient_id','latitude','longitude']]
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')



for lat, lon,city in zip(contactP_route['latitude'], contactP_route['longitude'], contactP_route['city']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('City: ' + str(city) + '<br>'),

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
p_info.head()
date_cols = ["confirmed_date", "released_date", "deceased_date"]

for col in date_cols:

    p_info[col] = pd.to_datetime(p_info[col])
p_info["time_to_release_since_confirmed"] = p_info["released_date"] - p_info["confirmed_date"]



p_info["time_to_death_since_confirmed"] = p_info["deceased_date"] - p_info["confirmed_date"]

p_info["duration_since_confirmed"] = p_info[["time_to_release_since_confirmed", "time_to_death_since_confirmed"]].min(axis=1)

p_info["duration_days"] = p_info["duration_since_confirmed"].dt.days

p_info["state_by_gender"] = p_info["state"] + "_" + p_info["sex"]
p_info.head()
plt.figure(figsize=(12, 8))

sns.boxplot(x="state",

            y="duration_days",

            order=["released", "deceased"],

            data=p_info)

plt.title("Time from confirmation to release or death", fontsize=16)

plt.xlabel("State", fontsize=16)

plt.ylabel("Days", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
order_duration_sex = ["female", "male"]

plt.figure(figsize=(12, 8))

sns.boxplot(x="sex",

            y="duration_days",

            order=order_duration_sex,

            hue="state",            

            hue_order=["released", "deceased"],

            data=p_info)

plt.title("Time from confirmation to release or death by gender",

          fontsize=16)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("Days", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
ageList = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '100s']

plt.figure(figsize=(12, 8))

sns.boxplot(x="age",

            y="duration_days",

            order=ageList,

            hue="state",

            hue_order=["released", "deceased"],

            data=p_info)

plt.title("Time from confirmation to release or death", fontsize=16)

plt.xlabel("Age Range", fontsize=16)

plt.ylabel("Days", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
timeGraph = time.set_index('date')
timeTemp = timeGraph[['confirmed', 'released', 'deceased']]
timeTemp.plot(figsize=(10,8))
confirm_perc=(time['confirmed'].sum()/(time['test'].sum()))*100

released_perc=(time['released'].sum()/(time['test'].sum()))*100

deceased_perc=(time['deceased'].sum()/(time['test'].sum()))*100
print("The percentage of confirm  is "+ str(confirm_perc) )

print("The percentage of released is "+ str(released_perc) )

print("The percentage of deceased is "+ str(deceased_perc) )
plt.figure(figsize=(100,50))

plt.bar(time.date, time.test,label="Test")

plt.bar(time.date, time.confirmed, label = "Confirmed")

plt.xlabel('Date')

plt.ylabel("Count")

plt.title('Test vs Confirmed',fontsize=100)

plt.legend(frameon=True, fontsize=12)

plt.show()
t_ageGraph = t_age.set_index('date')
t_ageGraph = t_ageGraph[['age', 'confirmed', 'deceased']]
t_ageGraph.head()
t_0s = t_ageGraph[t_ageGraph['age'] == '0s'][['confirmed', 'deceased']]

t_10s = t_ageGraph[t_ageGraph['age'] == '10s'][['confirmed', 'deceased']]

t_20s = t_ageGraph[t_ageGraph['age'] == '20s'][['confirmed', 'deceased']]

t_30s = t_ageGraph[t_ageGraph['age'] == '30s'][['confirmed', 'deceased']]

t_40s = t_ageGraph[t_ageGraph['age'] == '40s'][['confirmed', 'deceased']]

t_50s = t_ageGraph[t_ageGraph['age'] == '50s'][['confirmed', 'deceased']]

t_60s = t_ageGraph[t_ageGraph['age'] == '60s'][['confirmed', 'deceased']]

t_70s = t_ageGraph[t_ageGraph['age'] == '70s'][['confirmed', 'deceased']]

t_80s = t_ageGraph[t_ageGraph['age'] == '80s'][['confirmed', 'deceased']]
t_20s.plot(figsize=(8,8))
t_50s.plot(figsize=(8,8))
t_age = t_age[t_age['date'] == '2020-03-22']
t_age['confirmed'].sum()
ageList = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s']

ageConfirmed = pd.DataFrame(index=ageList, columns=['total_confirmed'])

ageDeceased = pd.DataFrame(index=ageList, columns=['total_deceased'])



for i in range(len(ageList)):

    ageConfirmed.loc[ageList[i]]['total_confirmed'] = t_age[t_age['age'] == ageList[i]]['confirmed'].sum()

    ageDeceased.loc[ageList[i]]['total_deceased'] = t_age[t_age['age'] == ageList[i]]['deceased'].sum()

    

ageConfirmed = ageConfirmed.sort_values(by='total_confirmed', ascending=True)

ageDeceased = ageDeceased.sort_values(by='total_deceased', ascending=True)
ax = ageConfirmed.plot.barh(figsize=(13,8))
plt.figure(figsize=(13, 8))

plt.title('Patients age')

p_info.age.value_counts(ascending=True).plot.barh()

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
ax = ageDeceased.plot.barh(figsize=(13,8))
t_genderGraph = t_gender.set_index('date')
t_genderGraph.head()
t_male = t_genderGraph[t_genderGraph['sex'] == 'male'][['confirmed', 'deceased']]

t_female = t_genderGraph[t_genderGraph['sex'] == 'female'][['confirmed', 'deceased']]
t_male.plot(figsize=(8,8))
t_female.plot(figsize=(8,8))
t_gender = t_gender[t_gender['date'] == '2020-03-22']
index = t_gender['sex'].unique()

sexConfirmed = pd.DataFrame(index=index, columns=['total_confirmed'])

sexDeceased = pd.DataFrame(index=index, columns=['total_deceased'])



for i in range(2):

    sexConfirmed.loc[index[i]]['total_confirmed'] = t_gender[t_gender['sex'] == index[i]]['confirmed'].sum()

    sexDeceased.loc[index[i]]['total_deceased'] = t_gender[t_gender['sex'] == index[i]]['deceased'].sum()
sexConfirmed
sexDeceased
sexConfirmed.plot.pie(y='total_confirmed', figsize=(5, 5))

plt.title("Male vs. Female")
sexDeceased.plot.pie(y='total_deceased', figsize=(5, 5))

plt.title("Male vs. Female")
t_provin = t_provin[t_provin['date'] == '2020-03-22']

t_provin = t_provin.reset_index(drop=True)
t_provin
provinceList = t_provin['province'].unique()
totalCProvince = pd.DataFrame(index=t_provin['province'].unique(), columns=['total_confirmed'])

totalDProvince = pd.DataFrame(index=t_provin['province'].unique(), columns=['total_deceased'])

totalRProvince = pd.DataFrame(index=t_provin['province'].unique(), columns=['total_released'])
for i in range(len(provinceList)):

    totalCProvince.loc[provinceList[i]]['total_confirmed'] = t_provin[t_provin['province'] == provinceList[i]]['confirmed'].sum()

    totalDProvince.loc[provinceList[i]]['total_deceased'] = t_provin[t_provin['province'] == provinceList[i]]['deceased'].sum()

    totalRProvince.loc[provinceList[i]]['total_released'] = t_provin[t_provin['province'] == provinceList[i]]['released'].sum()

    
totalCProvince
ax = totalCProvince.plot.barh(figsize=(12,8))
ax = totalDProvince.plot.barh(figsize=(12,8))
ax = totalRProvince.plot.barh(figsize=(12,8))
weather.head()
weatherTemp = weather.groupby(['date']).mean()
weatherTemp = weatherTemp[['avg_temp', 'precipitation', 'max_wind_speed', 'avg_relative_humidity']]
weatherTemp.head()
timeTemp = time[['date', 'confirmed']]

timeTemp = timeTemp.set_index('date')
weather_confirmed = pd.merge(timeTemp, weatherTemp, on='date')

weather_confirmed = weather_confirmed.reindex(columns = ['avg_temp', 'precipitation', 'max_wind_speed', 'avg_relative_humidity', 'confirmed'])

weather_confirmed.tail()
weather_confirmed.plot(figsize=(8,8))
df_korea = time[['date', 'confirmed']]
df_korea.tail()
# Make dataframe for Facebook Prophet prediction model

df_prophet = df_korea.rename(columns={

    'date': 'ds',

    'confirmed': 'y'

})



df_prophet.tail()
m = Prophet(

    changepoint_prior_scale=0.2, # increasing it will make the trend more flexible

    changepoint_range=0.98, # place potential changepoints in the first 98% of the time series

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)

m.fit(df_prophet)
future = m.make_future_dataframe(periods=7)

forecast = m.predict(future)

forecast.tail(7)
fig = m.plot(forecast)
df_korea = time[['date', 'confirmed']]

df_korea = df_korea[20:]

df_korea = df_korea.reset_index(drop=True)
df_korea_reg = df_korea.copy()
df_korea_reg = df_korea_reg.set_index('date')

df_korea_reg = df_korea_reg[20:]
df_korea_reg.index = pd.to_datetime(df_korea_reg.index)
x = np.arange(len(df_korea_reg)).reshape(-1, 1)

y = df_korea_reg.values
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)

_=model.fit(x, y)
test = np.arange(len(df_korea_reg)+7).reshape(-1, 1)

pred = model.predict(test)

prediction = pred.round().astype(int)

week = [df_korea_reg.index[0] + timedelta(days=i) for i in range(len(prediction))]

dt_idx = pd.DatetimeIndex(week)

predicted_count = pd.Series(prediction, dt_idx)
predicted_count.tail()
pd.plotting.register_matplotlib_converters()
df_korea_reg.plot()

predicted_count.plot()

plt.title('Prediction of Accumulated Confirmed Count')

plt.legend(['current confirmd count', 'predicted confirmed count'])

plt.show()
df_korea.tail()
model = ARIMA(df_korea['confirmed'].values, order=(1, 2, 1))

fit_model = model.fit(trend='c', full_output=True, disp=True)

fit_model.summary()
fit_model.plot_predict()

plt.title('Forecast vs Actual')

pd.DataFrame(fit_model.resid).plot()
forcast = fit_model.forecast(steps=7)

pred_y = forcast[0].tolist()

pd.DataFrame(pred_y)
t_provin = pd.read_csv(path+'TimeProvince.csv')
t_provin.head()
t_provinG = t_provin.groupby('date')['confirmed'].sum()

t_provinG = pd.DataFrame(t_provinG)
t_provinG['date'] = t_provinG.index

t_provinG.reset_index(drop=True, inplace=True)

t_provinG = t_provinG.reindex(columns=['date', 'confirmed'])
t_provinG['confirmed'] = t_provinG['confirmed'].diff()
t_provinG.head()
t_provinG.nlargest(3, 'confirmed')
t_provinG['confirmed'].sum()
t_provinG.plot(figsize=(10,8))
i = t_provin[((t_provin.province == 'Daegu') | (t_provin.province == 'Gyeongsangbuk-do'))].index

t_provin_except = t_provin.drop(i)
t_provin_exceptG = t_provin_except.groupby('date')['confirmed'].sum()

t_provin_exceptG = pd.DataFrame(t_provin_exceptG)
t_provin_exceptG['date'] = t_provin_exceptG.index

t_provin_exceptG.reset_index(drop=True, inplace=True)

t_provin_exceptG = t_provin_exceptG.reindex(columns=['date', 'confirmed'])
t_provin_exceptG['confirmed'] = t_provin_exceptG['confirmed'].diff()
t_provin_exceptG.head()
t_provin_exceptG['confirmed'].sum()
t_provin_exceptG.plot(figsize=(10,8))