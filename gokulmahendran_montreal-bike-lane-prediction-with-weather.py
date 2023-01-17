import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import datetime as datetime

import matplotlib as mpl

plt.style.use(['ggplot'])

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error
df=pd.read_csv("../input/montreal-bike-lanes/comptagesvelo2015.csv")
df.head()
def change_date_format(data):

    return datetime.datetime.strptime(data,"%d/%m/%Y")
df["dates"]=df["Date"].apply(change_date_format)
df["month"]=df["dates"].dt.month

df["day"]=df["dates"].dt.day

df["dow"]=df["dates"].dt.dayofweek
sns.heatmap(df.isnull())
mon=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]

plt.figure(figsize=(14,7))

plt.plot(df["Date"],df["Maisonneuve_1"],label="Maisonneuve_1")

plt.plot(df["Date"],df['Parc U-Zelt Test'],label='Parc U-Zelt Test')

plt.plot(df["Date"],df['Pont_Jacques_Cartier'],label='Pont_Jacques_Cartier')

plt.plot(df["Date"],df['Saint-Laurent U-Zelt Test'],label='Saint-Laurent U-Zelt Test')

plt.xticks(np.arange(1,319,30),mon)

plt.legend(loc=2,bbox_to_anchor=(1.05,1))
street_cols=df.columns[2:23]
len(street_cols)
street_cols
fig = plt.figure(figsize=(14, 26))

columns = 3

rows = 7

ax = []

for i in range(columns*rows):

    ax.append( fig.add_subplot(rows, columns, i+1) )

    ax[i].set_title(street_cols[i])

    plt.setp(ax,xticks=np.arange(1,319,30), xticklabels=mon)

for i in range(len(street_cols)):

    ax[i].plot(df["Date"],df[street_cols[i]],label=street_cols)



    



    
df[df["Maisonneuve_3"]==df["Maisonneuve_3"].max()]
df.fillna(value=0,inplace=True)
val=[]

for i in range(len(df)):

    val.append(df.iloc[i,2:23].values.sum())

    
df["total"]=val
plt.figure(figsize=(12,6))

plt.plot(df["Date"],df["total"])

plt.xticks(np.arange(1,319,30),mon)
df[df["total"]==df["total"].max()]
data=df[["month","dow","day","total"]]
plt.figure(figsize=(14,7))

sns.heatmap(data.corr(),annot=True)
data=df[["month","dow","day","total"]]
X=data.drop("total",axis=1)

y=data["total"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
pred=lr.predict(X_test)
plt.scatter(y_test,pred)
abr=AdaBoostRegressor(n_estimators=300)
abr.fit(X_train,y_train)
abr.score(X_test,y_test)
predictions=abr.predict(X_test)
plt.scatter(y_test,predictions)
weather=pd.read_csv("../input/weather-montreal-2015-en/eng-daily-01012015-12312015.csv")
weather.columns
plt.figure(figsize=(14,7))

sns.heatmap(weather.isnull())
weathers=weather[['Mean Temp (°C)','Total Rain (mm)','Total Snow (cm)','Total Precip (mm)','Snow on Grnd (cm)','Spd of Max Gust (km/h)']]
final_data=data.merge(weathers,left_index=True,right_index=True)
final_data.fillna(0,inplace=True)
sns.heatmap(final_data.isnull())
plt.figure(figsize=(14,7))

sns.heatmap(final_data.corr(),annot=True)
data=final_data.iloc[1]

types=[]

for i in range(len(final_data.columns)):

    types.append(type(data[i]))
columnss=pd.DataFrame(data=types,index=final_data.columns)
columnss
def change(data):

    if data=="<31":

        return 31

    else :

        return int(data)
final_data["Spd of Max Gust (km/h)"]=final_data["Spd of Max Gust (km/h)"].apply(change)
X=final_data.drop("total",axis=1)

y=final_data["total"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

lr.fit(X_train,y_train)
lr.score(X_test,y_test)
pred=lr.predict(X_test)
plt.scatter(y_test,pred)
abr=AdaBoostRegressor(n_estimators=300)
abr.fit(X_train,y_train)
abr.score(X_test,y_test)
predictions=abr.predict(X_test)
plt.scatter(y_test,predictions)