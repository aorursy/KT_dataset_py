import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

import geoplot
df = pd.read_csv("/kaggle/input/india-air-quality-data/data.csv", encoding = "ISO-8859-1")

df.head()
df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d') # date parse

df['year'] = df['date'].dt.year # year

df['year'] = df['year'].fillna(df["year"].min())

df['year'] = df['year'].values.astype(int)
print (df.get_dtype_counts())
def printNullValues(df):

    total = df.isnull().sum().sort_values(ascending = False)

    total = total[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = total / len(df) * 100

    percent = percent[df.isnull().sum().sort_values(ascending = False) != 0]

    concat = pd.concat([total, percent], axis=1, keys=['Total','Percent'])

    print (concat)

    print ( "-------------")
printNullValues(df)
df["type"].value_counts()


sns.catplot(x = "type", kind = "count",  data = df, height=5, aspect = 4)
grp = df.groupby(["type"]).mean()["so2"].to_frame()

grp.plot.bar(figsize = (20,10))
grp = df.groupby(["type"]).mean()["no2"].to_frame()

grp.plot.bar(figsize = (20,10))


df[['so2', 'state']].groupby(['state']).median().sort_values("so2", ascending = False).plot.bar(figsize=(20,10))

df[['so2','year','state']].groupby(["year"]).median().sort_values(by='year',ascending=False).plot(figsize=(20,10))


df[['no2', 'state']].groupby(['state']).median().sort_values("no2", ascending = False).plot.bar(figsize=(20,10))

df[['no2','year','state']].groupby(["year"]).median().sort_values(by='year',ascending=False).plot(figsize=(20,10))


df[['spm', 'state']].groupby(['state']).median().sort_values("spm", ascending = False).plot.bar(figsize=(20,10))

df[['spm','year','state']].groupby(["year"]).median().sort_values(by='year',ascending=False).plot(figsize=(20,10))
fig, ax = plt.subplots(figsize=(20,10))      

sns.heatmap(df.pivot_table('so2', index='state',columns=['year'],aggfunc='median',margins=True),ax = ax,annot=True, linewidths=.5)
fig, ax = plt.subplots(figsize=(20,10))      

sns.heatmap(df.pivot_table('no2', index='state',columns=['year'],aggfunc='median',margins=True),ax = ax,annot=True, linewidths=.5)
fig, ax = plt.subplots(figsize=(20,10))      

sns.heatmap(df.pivot_table('spm', index='state',columns=['year'],aggfunc='median',margins=True),ax = ax,annot=False, linewidths=.5)
temp = df.pivot_table('so2', index='year',columns=['state'],aggfunc='median',margins=True).reset_index()

temp = temp.drop("All", axis = 1)

temp = temp.set_index("year")

temp.plot(figsize=(20,10))
temp = df.pivot_table('no2', index='year',columns=['state'],aggfunc='median',margins=True).reset_index()

temp = temp.drop("All", axis = 1)

temp = temp.set_index("year")

temp.plot(figsize=(20,10))
temp = df.pivot_table('spm', index='year',columns=['state'],aggfunc='median',margins=True).reset_index()

temp = temp.drop("All", axis = 1)

temp = temp.set_index("year")

temp.plot(figsize=(20,10))
india = gpd.read_file('/kaggle/input/maps-of-india/India_SHP/INDIA.shp')

india.info()
india.plot()
india["ST_NAME"] = india["ST_NAME"].apply(lambda x: x.lower())



india = india.set_index("ST_NAME")



df["state"] = df["state"].apply(lambda x: x.lower())
df_before_2000 = df[df["year"] < 2000]

df_before_2000 = df_before_2000.groupby("state").mean()
df_after_2000 = df[df["year"] > 2000]

df_after_2000 = df_after_2000.groupby("state").mean()
result = pd.concat([df_before_2000, india], axis=1, sort=False)

result = result [result["geometry"] != None]

result = result [result["year"] > 0]

from geopandas import GeoDataFrame

crs = {'init': 'epsg:4326'}

gdf = GeoDataFrame(result, crs=crs, geometry=result ["geometry"])

gdf['centroid'] = gdf.geometry.centroid

fig,ax = plt.subplots(figsize=(20,10))

gdf.plot(column='so2',ax=ax,alpha=0.4,edgecolor='black',cmap='cool', legend=True)

plt.title("Mean So2 before 2000")

plt.axis('off')



for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.index):

    ax.annotate(label, xy=(x, y), xytext=(3,3), textcoords="offset points",color='gray')
result = pd.concat([df_after_2000, india], axis=1, sort=False)

result = result [result["geometry"] != None]

result = result [result["year"] > 0]

from geopandas import GeoDataFrame

crs = {'init': 'epsg:4326'}

gdf = GeoDataFrame(result, crs=crs, geometry=result ["geometry"])

gdf['centroid'] = gdf.geometry.centroid

fig,ax = plt.subplots(figsize=(20,10))

gdf.plot(column='so2',ax=ax,alpha=0.4,edgecolor='black',cmap='cool', legend=True)

plt.title("Mean So2 after 2000")

plt.axis('off')



for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.index):

    ax.annotate(label, xy=(x, y), xytext=(3,3), textcoords="offset points",color='gray')
df_so2 = df[["date", "so2"]]

df_so2 = df_so2.set_index("date")

df_so2 = df_so2.dropna()
df_so2_resample = df_so2.resample(rule = "M").mean().ffill()
df_so2_resample.plot(figsize = (20,10))
df_so2_resample["so2"].resample("A").mean().plot.bar(figsize = (20,10))
df_so2_resample.plot(figsize = (20,10))

df_so2_resample.rolling(window = 7).mean()["so2"].plot(figsize = (20,10))
df_so2_resample["EWMA-7"] = df_so2_resample["so2"].ewm(span=7).mean()
df_so2_resample.plot(figsize = (20,10))
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df_so2_resample["so2"], model = "multiplicative") 
fig = result.plot()
from statsmodels.tsa.stattools import adfuller

result = adfuller(df_so2_resample["so2"])

print('Augmented Dickey-Fuller Test:')

labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']



for value,label in zip(result,labels):

    print(label+' : '+str(value) )

    

if result[1] <= 0.05:

    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")

else:

    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
df_so2_resample["so2_first_diff"] = df_so2_resample["so2"] - df_so2_resample["so2"].shift(7)

# CHECK

result = adfuller(df_so2_resample["so2_first_diff"].dropna() )

print('Augmented Dickey-Fuller Test:')

labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']



for value,label in zip(result,labels):

    print(label+' : '+str(value) )

    

if result[1] <= 0.05:

    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")

else:

    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
df_so2_resample["so2_first_diff"].plot(figsize = (20,10))
df_so2_resample["so2_second_diff"] = df_so2_resample["so2_first_diff"] - df_so2_resample["so2_first_diff"].shift(7)

df_so2_resample["so2_second_diff"].plot(figsize = (20,10))
import statsmodels.api as sm



model = sm.tsa.statespace.SARIMAX(df_so2_resample["so2"],order=(0,1,0), seasonal_order=(1,1,1,48))

results = model.fit()

print(results.summary())

results.resid.plot()
results.resid.plot(kind='kde')
df_so2_resample['forecast'] = results.predict(start = 250, end= 400, dynamic= True)  

df_so2_resample[['so2','forecast']].plot(figsize=(20,10))
from pandas.tseries.offsets import DateOffset

future_dates = [df_so2_resample.index[-1] + DateOffset(months=x) for x in range(0,24) ]

future_dates_df = pd.DataFrame(index=future_dates[1:],columns=df_so2_resample.columns)

future_df = pd.concat([df_so2_resample,future_dates_df])

future_df['forecast2'] = results.predict(start = 348, end = 540, dynamic= True)  

future_df[['so2', 'forecast2']].plot(figsize=(20, 10)) 