import numpy as np 

from matplotlib import font_manager, rc

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator 

plt.style.use('seaborn')

%matplotlib inline 

import plotly.express as px

import plotly.offline as py



# font

import os

plt.rcParams['axes.unicode_minus'] = False

fontpath = "../input/nanum-plot/NanumGothic.ttf"

fontprop = font_manager.FontProperties(fname=fontpath,size=12)
#print(os.listdir("../input/nanum-plot/NanumGothic.ttf"))
time_age_raw = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv')

region_raw = pd.read_csv('/kaggle/input/coronavirusdataset/Region.csv')

time_raw = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')

weather_raw = pd.read_csv('/kaggle/input/coronavirusdataset/Weather.csv')

search_trend_raw = pd.read_csv('/kaggle/input/coronavirusdataset/SearchTrend.csv')

time_province_raw = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')

time_gender_raw = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')

patient_info_raw = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')

patient_route_raw = pd.read_csv('/kaggle/input/coronavirusdataset/PatientRoute.csv')

seoul_floating_raw = pd.read_csv('/kaggle/input/coronavirusdataset/SeoulFloating.csv')

case_raw = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')
time_age_raw
time_age_raw.tail(10)
time_age_raw['age'].unique()
# create a color palette

palette = plt.get_cmap('Set1')

 

# multiple line plot

num=0

for ages in time_age_raw['age'].unique():

    num+=1

    plt.plot(time_age_raw.loc[time_age_raw['age']==ages, 'date'], time_age_raw.loc[time_age_raw['age']==ages, 'confirmed'], marker='', color=palette(num), linewidth=1, alpha=0.9, label=ages)

 

    # Add legend

    plt.legend(loc=2, ncol=2)



    # Add titles

    plt.title("연령별 확진자 수 추이", loc='left', fontsize=12, fontweight=0, color='orange')

    plt.xlabel("Time")

    plt.ylabel("confirmed")

    plt.xticks(size=20, rotation=90)
region_raw
time_raw
weather_raw
search_trend_raw
time_province_raw
time_gender_raw
patient_info_raw
patient_route_raw
seoul_floating_raw
case_raw
time = time_raw.copy()

time.index = time['date']

confirmed = time['confirmed']

deaths= time['deceased']

recoveries = time['released']
dates = confirmed.keys()

korea_cases = []

total_deaths = [] 

mortality_rate = []

recovery_rate = [] 

total_recovered = [] 

total_active = [] 



for i in dates:

    confirmed_sum = confirmed[i]

    death_sum = deaths[i]

    recovered_sum = recoveries[i]

    

    # confirmed, deaths, recovered, and active

    korea_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    total_recovered.append(recovered_sum)

    total_active.append(confirmed_sum-death_sum-recovered_sum)

    

    # calculate rates

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)
patient_route = patient_route_raw.copy()

patient_route = patient_route.rename(columns={"province": "province_route", "city": "city_route", "date": "date_route","latitude":"latitude_route", "longitude":"longitude_route"})



region = region_raw.copy()

region = region.rename(columns={"code": "code_region"})



patient_raw = pd.merge(patient_info_raw, patient_route, on=['patient_id', 'global_num'], how='left')

patient_raw = pd.merge(patient_info_raw, region, on=['province', 'city'], how='left')



date_province_raw = pd.merge(time_province_raw, region, on=['province'], how='left')
patient_raw
date_province_raw
plt.figure(figsize=(16, 9))

plt.plot(dates, korea_cases)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 20})

plt.xticks(size=20, rotation=90)

plt.yticks(size=20)

plt.show()
df = time_province_raw.tail(17)



plt.figure(figsize=(16, 9))

plt.barh(df['province'], df['confirmed'])

plt.title('# of Covid-19 Confirmed Cases in Provinces', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
confirmed[-1]
time_raw.index = time_raw['date']

confirmed = time_raw['confirmed']

deaths= time_raw['deceased']

recoveries = time_raw['released']



count = [confirmed[-1], deaths[-1], recoveries[-1]]

state = ['confired', 'deaths',' recoveries']



states = pd.DataFrame()

states["count"] = count

states["status"] =  state

fig = px.pie(states,

             values="count",

             names="status",

             title="Current state of patients",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")

fig.show()
fig = px.pie( values=patient_route_raw.groupby(['city']).size().values,names=patient_route_raw.groupby(['city']).size().index)

fig.update_layout(

    font=dict(

        size=15,

        color="#242323"

    )

    )   

    

py.iplot(fig)
mean_recovery_rate = np.mean(recovery_rate)

plt.figure(figsize=(16, 9))

plt.plot(dates, recovery_rate, color='orange')

plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')

plt.title('Recovery Rate of Coronavirus Over Time', size=30)

plt.legend(['recovery Rate', 'mean recovery rate ='+str(round(mean_recovery_rate, 4))], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Recovery Rate', size=30)

plt.xticks(size=20, rotation=90)

plt.yticks(size=20)

plt.show()
mean_mortality_rate = np.mean(mortality_rate)

plt.figure(figsize=(16, 9))

plt.plot(dates, mortality_rate, color='orange')

plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')

plt.title('Mortality Rate of Coronavirus Over Time', size=30)

plt.legend(['mortality rate', 'mean mortality rate='+str(round(mean_mortality_rate, 4))], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Mortality Rate', size=30)

plt.xticks(size=20, rotation=90)

plt.yticks(size=20)

plt.show()