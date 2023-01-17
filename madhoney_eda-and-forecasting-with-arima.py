import numpy as np

import statsmodels

import statsmodels.api as sm

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

sns.set_style("darkgrid")

import sklearn

from scipy import stats

import pyproj

from mpl_toolkits.basemap import Basemap

from sklearn.preprocessing import LabelEncoder

from itertools import product
gtb_df = pd.read_csv("../input/globalterrorismdb_0617dist.csv",delimiter=",",encoding="latin1",error_bad_lines=False)
gtb_df.head()
def only_one(array,null_info,is_true = True):

    result = []

    for item,inf in zip(array,null_info):

        if inf == is_true:

            result.append(item)

    return pd.Index(result)
null_info = gtb_df.isnull().any()

columns = gtb_df.columns

no_null = only_one(columns,null_info,False)
time_series_success = pd.pivot_table(index=["iyear","imonth"], 

                                     values="success", data = gtb_df, aggfunc=np.sum)

time_series_count = pd.pivot_table(index=["iyear","imonth"],

                                   values="success", data = gtb_df, aggfunc= lambda x: len(x))

time_series = time_series_success.merge(time_series_count,how="inner",

                                        left_index=True,right_index=True,suffixes=["","count"])
time_series.head(12)
time_series.plot(figsize=(20,6))

plt.title("Success and nosuccess attack through time")
geo_data = gtb_df[["latitude","longitude","success"]]

europe = geo_data[np.logical_and(geo_data["longitude"]>-10,geo_data["longitude"]<50)]

europe = europe[np.logical_and(europe["latitude"]>25,europe["latitude"]<55)]
plt.figure(figsize=(15,15))

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)

x,y = m(list(geo_data["longitude"].values),list(geo_data["latitude"].values))

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary(fill_color='aqua')

m.plot(x, y,'go',markersize=2,alpha=0.4)

plt.title("Terrorism around the world")

plt.show()
plt.figure(figsize=(15,15))

m = Basemap(projection='mill',llcrnrlat=25,urcrnrlat=55, llcrnrlon=-10,urcrnrlon=50,lat_ts=20,resolution='c',lat_0=True,lat_1=True)

x,y = m(list(europe["longitude"].values),list(europe["latitude"].values))

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary(fill_color='aqua')

m.drawcountries()

m.plot(x, y,'go',markersize=2,alpha=0.4)

plt.title("Terrorism in Europe")

plt.show()
geo_data["attack"] = gtb_df["attacktype1_txt"]

geo_data.head()

set(geo_data["attack"])
sns.countplot(geo_data["attack"])

plt.xticks(rotation="vertical")

plt.title("Attack types count")
country_counts = gtb_df["country_txt"].value_counts()

sns.barplot(country_counts.index[:10],country_counts.values[:10])

plt.xticks(rotation = "vertical")
#cause python has no inverse box-cox

def invcoxbox(y,lambd):

    if lambd == 0:

        return np.exp(y)

    else:

        return np.exp(np.log(lambd*y+1)/lambd)
#creating datetime index for decomposition and Diki-Fuller

date_list = []

for year,month in zip(gtb_df["iyear"].values,gtb_df["imonth"].values):

    date_list.append(str(month+1)+"/1/"+str(year))
gtb_df["date"] = np.array(date_list)

gtb_df["date"] = pd.to_datetime(gtb_df["date"])

gtb_df["date"].isnull().any()

time_series = pd.pivot_table(values="success",index="date",aggfunc=np.sum,data=gtb_df)
time_series.plot(figsize=(15,6))

plt.ylabel("Terrorism attacks")

plt.title("Terrorism attacks count by time")
#STL decomposition

plt.figure(figsize=(20,12))

sm.tsa.seasonal_decompose(time_series,freq=30).plot()

print("Критерий Дики-Фуллера:",sm.tsa.stattools.adfuller(time_series.success)[1])
seasonal = sm.tsa.seasonal_decompose(time_series,freq=30).seasonal

plt.figure(figsize=(20,12))

plt.plot(seasonal)

plt.xlim(["1/1/1997","1/1/2005"])
#box-cox scaling to reduce variance

time_series["success_boxcox"],lmbd = stats.boxcox(time_series.success)

plt.figure(figsize=(20,12))

time_series.success_boxcox.plot()

print("Best Lambda parameter:",lmbd)

print("Критерий Дики-Фуллер после преобразования бокса-кокса:",sm.tsa.stattools.adfuller(time_series.success_boxcox)[1])
time_series.head()
time_series["success_box_diff"] = time_series["success_boxcox"] - time_series["success_boxcox"].shift(30)

time_series = time_series.fillna(0)

sm.tsa.seasonal_decompose(time_series.success_box_diff,freq=30).plot()

print("Критерий Дики-Фуллера после сезонного дифиренцирования:",sm.tsa.stattools.adfuller(time_series.success_box_diff)[1])
time_series["success_box_diff"] = time_series["success_box_diff"] - time_series["success_box_diff"].shift(1)

time_series = time_series.fillna(0)

sm.tsa.seasonal_decompose(time_series.success_box_diff,freq=30).plot()

print("Критерий Дики-Фуллера:",sm.tsa.stattools.adfuller(time_series.success_box_diff)[1])
plt.figure(figsize=(20,16))

ax = plt.subplot(211)

sm.graphics.tsa.plot_acf(time_series.success_box_diff[31:].values.squeeze(),lags=120,ax=ax)

plt.show()

plt.figure(figsize=(20,16))

ax = plt.subplot(212)

sm.graphics.tsa.plot_pacf(time_series.success_box_diff[31:].values.squeeze(),lags=120,ax=ax)

plt.show()
Q = range(3)

q = range(2)

P = range(3)

p = range(2)

d = 1

D = 1

parameters = list(product(p,q,P,Q))

len(parameters)
%%time

results = []

best_aic = float("inf")

for param in parameters:

    try:

        model = sm.tsa.statespace.SARIMAX(time_series.success_boxcox, order=(param[0],d,param[1]),

                                          seasonal_order=(param[2],D,param[3],30)).fit(disp=-1)

    except:

        print("Wrong params:",param)

        continue

    aic = model.aic

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results.append([param,model.aic])
result_table = pd.DataFrame(results)

result_table.columns = ["params","aic"]

result_table.sort_values(by="aic",ascending=[True]).head()
best_model = sm.tsa.statespace.SARIMAX(time_series.success_boxcox, order=(0,1,1),

                                       seasonal_order=(1,1,1,30)).fit(disp=-1)
best_model.summary()
time_series["model"] = invcoxbox(best_model.fittedvalues,lmbd)

plt.figure(figsize=(20,12))

time_series.success.plot()

time_series.model[30:].plot()

plt.title("True data vs predicted data")
time_series.tail(4)
to_predict = []

for i in range(5):

    for j in range(12):

        year = str(2016+i)

        month = str(j+1)

        to_predict.append(month+"/1/"+year)

to_predict = pd.to_datetime(to_predict)

forecast = pd.DataFrame(index=to_predict,columns=["forecast"])

forecast.tail()

start = 563-12

end = start+60-1
forecast["forecast"]=invcoxbox(best_model.predict(start=start,end=end).values,lmbd)
forecast.head()
plt.figure(figsize=(20,12))

plt.plot(time_series.success)

plt.plot(forecast.forecast)