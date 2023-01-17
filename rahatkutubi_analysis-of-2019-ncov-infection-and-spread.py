# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
np.set_printoptions(precision=6, suppress=True)
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
df_C = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df_R = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
df_D = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

region_metadata_df = pd.read_csv("/kaggle/input/covid19-forecasting-metadata/region_metadata.csv")
region_date_metadata_df = pd.read_csv("/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv")

df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
df_table = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])

# Retriving Dataset
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_C_2=df_C.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_C_global_sum=df_C_2.sum()
df_D_2=df_D.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_D_global_sum=df_D_2.sum()
df_R_2=df_R.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_R_global_sum=df_R_2.sum()
df_A_global_sum=df_C_global_sum-df_D_global_sum-df_R_global_sum
dates=df_C_2.keys()
dates 
since_date_1_22 = np.array([ a for a in range(len(dates))])
since_date_1_22
df_R_global_sum
df_C_2
plt.figure(figsize=(12,8))
plt.title("Global Confirmed case spread over time",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Confirmed Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_C_global_sum)
plt.plot(since_date_1_22,df_D_global_sum)
plt.plot(since_date_1_22,df_R_global_sum)
plt.plot(since_date_1_22,df_A_global_sum)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_world_meter = pd.read_csv("/kaggle/input/worldmeters-corona-dataset/worldmeters_corona_dataset.csv", encoding= 'unicode_escape')
df_world_meter.fillna(0, inplace=True)

df_world_meter.keys()

df_world_meter['test_per_m']
df_world_meter.test_per_m=pd.to_numeric(df_world_meter['test_per_m'].str.replace(',','')).fillna(0).astype(np.int64)

df_C.loc[df_C['Country/Region'] == "US", "Country/Region"] = "USA"
df_D.loc[df_D['Country/Region'] == "US", "Country/Region"] = "USA"
df_R.loc[df_R['Country/Region'] == "US", "Country/Region"] = "USA"

df_C.loc[df_C['Country/Region'] == "UK", "Country/Region"] = "USA"
df_D.loc[df_D['Country/Region'] == "UK", "Country/Region"] = "USA"
df_R.loc[df_R['Country/Region'] == "UK", "Country/Region"] = "USA"

df_C.loc[df_C['Country/Region']  == 'Korea, South', "Country/Region"] = 'South Korea'
df_D.loc[df_D['Country/Region']  == 'Korea, South', "Country/Region"] = 'South Korea'
df_R.loc[df_R['Country/Region']  == 'Korea, South', "Country/Region"] = 'South Korea'

df_C.loc[df_confirmed['Country/Region'] == 'Taiwan*', "Country/Region"] = 'Taiwan'
df_D.loc[df_D['Country/Region'] == 'Taiwan*', "Country/Region"] = 'Taiwan'
df_R.loc[df_R['Country/Region']  == "Taiwan*", "Country/Region"] = "Taiwan"



df_C.loc[df_C['Country/Region'] =='Taiwan*', "Country/Region"] = 'Taiwan'
df_D.loc[df_D['Country/Region'] == 'Taiwan*', "Country/Region"] = 'Taiwan'
df_R.loc[df_R['Country/Region'] == "Taiwan*", "Country/Region"] = "Taiwan"

df_C.loc[df_C['Country/Region'] == 'Congo (Kinshasa)', "Country/Region"] = 'Democratic Republic of the Congo'
df_D.loc[df_D['Country/Region'] == 'Congo (Kinshasa)', "Country/Region"] = 'Democratic Republic of the Congo'
df_R.loc[df_R['Country/Region']  == "Congo (Kinshasa)", "Country/Region"] = "Democratic Republic of the Congo"


df_C.loc[df_C['Country/Region'] == "Cote d'Ivoire", "Country/Region"] = "Côte d'Ivoire"
df_D.loc[df_D['Country/Region'] == "Cote d'Ivoire", "Country/Region"] = "Côte d'Ivoire"
df_R.loc[df_R['Country/Region']  == "Cote d'Ivoire", "Country/Region"] = "Côte d'Ivoire"


df_world_meter.loc[df_world_meter['Country'] == "UK", "Country"] = "United Kingdom"

df_world_meter['Country'].values
high_test_list=df_world_meter.loc[df_world_meter['test_per_m']>20000 , 'Country']
midium_test_list=df_world_meter.loc[(df_world_meter['test_per_m']<20000) & (df_world_meter['test_per_m']>5000) , 'Country']
low_test_list=df_world_meter.loc[(df_world_meter['test_per_m']<5000) & (df_world_meter['test_per_m']>1500)  , 'Country']
vlow_test_list=df_world_meter.loc[(df_world_meter['test_per_m']<1000) & (df_world_meter['test_per_m']>50)  , 'Country']
high_test_list.values
midium_test_list.values
low_test_list.values
vlow_test_list.values
df_C_ht=df_C.loc[df_C['Country/Region'].isin(high_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_C_global_sum_ht=df_C_ht.sum()
df_D_ht=df_D.loc[df_D['Country/Region'].isin(high_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_D_global_sum_ht=df_D_ht.sum()
df_R_ht=df_R.loc[df_R['Country/Region'].isin(high_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_R_global_sum_ht=df_R_ht.sum()
df_A_global_sum_ht=df_C_global_sum_ht-df_D_global_sum_ht-df_R_global_sum_ht
df_C_mt=df_C.loc[df_C['Country/Region'].isin(midium_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_C_global_sum_mt=df_C_mt.sum()
df_D_mt=df_D.loc[df_D['Country/Region'].isin(midium_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_D_global_sum_mt=df_D_mt.sum()
df_R_mt=df_R.loc[df_R['Country/Region'].isin(midium_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_R_global_sum_mt=df_R_mt.sum()
df_A_global_sum_mt=df_C_global_sum_mt-df_D_global_sum_mt-df_R_global_sum_mt
df_C_lt=df_C.loc[df_C['Country/Region'].isin(low_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_C_global_sum_lt=df_C_lt.sum()
df_D_lt=df_D.loc[df_D['Country/Region'].isin(low_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_D_global_sum_lt=df_D_lt.sum()
df_R_lt=df_R.loc[df_R['Country/Region'].isin(low_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_R_global_sum_lt=df_R_lt.sum()
df_A_global_sum_lt=df_C_global_sum_lt-df_D_global_sum_lt-df_R_global_sum_lt
df_C_vlt=df_C.loc[df_C['Country/Region'].isin(vlow_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_C_global_sum_vlt=df_C_vlt.sum()
df_D_vlt=df_D.loc[df_D['Country/Region'].isin(vlow_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_D_global_sum_vlt=df_D_vlt.sum()
df_R_vlt=df_R.loc[df_R['Country/Region'].isin(vlow_test_list.values)].copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_R_global_sum_vlt=df_R_vlt.sum()
df_A_global_sum_vlt=df_C_global_sum_vlt-df_D_global_sum_vlt-df_R_global_sum_vlt
df_C_global_sum_vlt
df_A_global_sum_vlt
plt.figure(figsize=(12,8))
plt.title("Global spread over time for high test rate Countries",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_C_global_sum_ht)
plt.plot(since_date_1_22,df_D_global_sum_ht)
plt.plot(since_date_1_22,df_R_global_sum_ht)
plt.plot(since_date_1_22,df_A_global_sum_ht)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()

plt.figure(figsize=(12,8))
plt.title("Global spread over time for midium test rate Countries",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_C_global_sum_mt)
plt.plot(since_date_1_22,df_D_global_sum_mt)
plt.plot(since_date_1_22,df_R_global_sum_mt)
plt.plot(since_date_1_22,df_A_global_sum_mt)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()

plt.figure(figsize=(12,8))
plt.title("Global spread over time for low test rate Countries",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_C_global_sum_lt)
plt.plot(since_date_1_22,df_D_global_sum_lt)
plt.plot(since_date_1_22,df_R_global_sum_lt)
plt.plot(since_date_1_22,df_A_global_sum_lt)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()


plt.figure(figsize=(12,8))
plt.title("Global spread over time for very low test rate Countries",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_C_global_sum_vlt)
plt.plot(since_date_1_22,df_D_global_sum_vlt)
plt.plot(since_date_1_22,df_R_global_sum_vlt)
plt.plot(since_date_1_22,df_A_global_sum_vlt)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()

df_c = df_C[df_C['Country/Region']=='Italy']
df_c = df_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_c=df_c.transpose()

df_d = df_D[df_D['Country/Region']=='Italy']
df_d = df_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_d=df_d.transpose()

df_r = df_R[df_R['Country/Region']=='Italy']
df_r = df_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_r.index+=6
df_r=df_r.transpose()
df_a=df_c-df_d-df_r
plt.figure(figsize=(12,8))
plt.title("Global spread over time for Itally",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_c)
plt.plot(since_date_1_22,df_d)
plt.plot(since_date_1_22,df_r)
plt.plot(since_date_1_22,df_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_c = df_C[df_C['Country/Region']=='Spain']
df_c = df_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_c=df_c.transpose()

df_d = df_D[df_D['Country/Region']=='Spain']
df_d = df_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_d=df_d.transpose()

df_r = df_R[df_R['Country/Region']=='Spain']
df_r = df_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_r.index+=2
df_r=df_r.transpose()
df_a=df_c-df_d-df_r


plt.figure(figsize=(12,8))
plt.title("Global spread over time for Spain",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_c)
plt.plot(since_date_1_22,df_d)
plt.plot(since_date_1_22,df_r)
plt.plot(since_date_1_22,df_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_c
df_c = df_C[df_C['Country/Region']=='USA']
df_c = df_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_c=df_c.transpose()

df_d = df_D[df_D['Country/Region']=='USA']
df_d = df_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_d=df_d.transpose()

df_r = df_R[df_R['Country/Region']=='USA']
df_r = df_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_r.index+=0
df_r=df_r.transpose()
df_a=df_c-df_d-df_r


plt.figure(figsize=(12,8))
plt.title("Global spread over time for United State",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_c)
plt.plot(since_date_1_22,df_d)
plt.plot(since_date_1_22,df_r)
plt.plot(since_date_1_22,df_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()


df_c = df_C[df_C['Country/Region']=='United Kingdom']
df_c = df_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_c=df_c.sum()

df_d = df_D[df_D['Country/Region']=='United Kingdom']
df_d = df_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_d=df_d.sum()

df_r = df_R[df_R['Country/Region']=='United Kingdom']
df_r = df_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_r=df_r.sum()
df_a=df_c-df_d-df_r


plt.figure(figsize=(12,8))
plt.title("Global spread over time for United Kingdom",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_c)
plt.plot(since_date_1_22,df_d)
plt.plot(since_date_1_22,df_r)
plt.plot(since_date_1_22,df_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_italy_c = df_C[df_C['Country/Region']=='South Korea']
df_italy_c = df_italy_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_italy_c=df_italy_c.transpose()

df_italy_d = df_D[df_D['Country/Region']=='South Korea']
df_italy_d = df_italy_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_italy_d=df_italy_d.transpose()

df_italy_r = df_R[df_R['Country/Region']=='South Korea']
df_italy_r = df_italy_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_italy_r.index+=6
df_italy_r=df_italy_r.transpose()
df_italy_a=df_italy_c-df_italy_d-df_italy_r


plt.figure(figsize=(12,8))
plt.title("Global spread over time for South Korea",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_italy_c)
plt.plot(since_date_1_22,df_italy_d)
plt.plot(since_date_1_22,df_italy_r)
plt.plot(since_date_1_22,df_italy_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_italy_c = df_C[df_C['Country/Region']=='Bangladesh']
df_italy_c = df_italy_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_italy_c=df_italy_c.transpose()

df_italy_d = df_D[df_D['Country/Region']=='Bangladesh']
df_italy_d = df_italy_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_italy_d=df_italy_d.transpose()

df_italy_r = df_R[df_R['Country/Region']=='Bangladesh']
df_italy_r = df_italy_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_italy_r.index+=0
df_italy_r=df_italy_r.transpose()
df_italy_a=df_italy_c-df_italy_d-df_italy_r


plt.figure(figsize=(12,8))
plt.title("Global spread over time for Bangladesh",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_italy_c)
plt.plot(since_date_1_22,df_italy_d)
plt.plot(since_date_1_22,df_italy_r)
plt.plot(since_date_1_22,df_italy_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_c = df_C[df_C['Country/Region']=='India']
df_c = df_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_c=df_c.sum()

df_d = df_D[df_D['Country/Region']=='India']
df_d = df_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_d=df_d.sum()

df_r = df_R[df_R['Country/Region']=='India']
df_r = df_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_r=df_r.sum()
df_a=df_c-df_d-df_r


plt.figure(figsize=(12,8))
plt.title("Global spread over time for India",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_c)
plt.plot(since_date_1_22,df_d)
plt.plot(since_date_1_22,df_r)
plt.plot(since_date_1_22,df_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_c = df_C[df_C['Country/Region']=='Brazil']
df_c = df_c.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
#df_C_global_sum_mt=df_C_mt.sum()
df_c=df_c.sum()

df_d = df_D[df_D['Country/Region']=='Brazil']
df_d = df_d.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_d=df_d.sum()

df_r = df_R[df_R['Country/Region']=='Brazil']
df_r = df_r.copy().drop(columns=['Country/Region','Province/State','Lat', 'Long'])
df_r=df_r.sum()
df_a=df_c-df_d-df_r


plt.figure(figsize=(12,8))
plt.title("Global spread over time for Brazil",fontsize=24)
plt.xlabel("Since date 1/22",fontsize=24)
plt.ylabel('# of global Case', fontsize=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.plot(since_date_1_22,df_c)
plt.plot(since_date_1_22,df_d)
plt.plot(since_date_1_22,df_r)
plt.plot(since_date_1_22,df_a)
plt.legend(['Confirmed','Deaths','Recover', 'Active'], prop={'size':23})
plt.show()
df_R.keys()
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
future_days=10
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
future_forcast= np.array([i for i in range(len(dates))+future_days]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
dates = df_C_global_sum.keys()
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
future_days=10
future_forcast= np.array([i for i in range((len(dates))+future_days)]).reshape(-1, 1)
df_Confirmed= np.array([df_C_global_sum[dates[i]] for i in range(len(dates))]).reshape(-1, 1)
#df_Confirmed=df_C_global_sum.copy().drop(df_C_global_sum.columns[i], axis=1)
print(df_Confirmed)

X_train_c,X_test_c, y_train_c, y_test_c = train_test_split(days_since_1_22[:],df_Confirmed[:],test_size=0.12, shuffle=False)
#y_train_c=[x[0] for x in y_train_c]
#y_test_c=[x[0] for x in y_test_c]
y_train_c
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_c)
poly_X_test_confirmed = poly.fit_transform(X_test_c)
poly_future_forcast = poly.fit_transform(future_forcast)
poly_X_train_confirmed
# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_c)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_c))
print('MSE:',mean_squared_error(test_linear_pred, y_test_c))
print(linear_model.coef_)

plt.plot(y_test_c)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])
X_train_c,X_test_c, y_train_c, y_test_c = train_test_split(days_since_1_22[60:],df_Confirmed[60:],test_size=0.12, shuffle=False)
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_c)
poly_X_test_confirmed = poly.fit_transform(X_test_c)
poly_future_forcast = poly.fit_transform(future_forcast[60:])
# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_c)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_c))
print('MSE:',mean_squared_error(test_linear_pred, y_test_c))
plt.plot(y_test_c)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])
linear_pred.shape

plt.plot(linear_pred)
plt.legend(['Polynomial Regression Predictions'])
linear_pred[77]