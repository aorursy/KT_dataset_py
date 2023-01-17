import pandas as pd
import numpy as np
import datetime
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import os
#print(os.listdir('../input'))
df=pd.read_csv("../input/recorridos-realizados-2018.CSV", encoding='latin-1')
print('There are %d bikes released' % len(df))

df.head()
df.tail()
df.rename(columns={'bici_id_usuario': 'id_user',
                   'bici_Fecha_hora_retiro': 'date_released',
                   'bici_tiempo_uso': 'time_use',
                   'bici_nombre_estacion_origen': 'station_name_released',
                   'bici_estacion_origen': 'station_id_released',
                   'bici_nombre_estacion_destino': 'station_name_lock',
                   'bici_estacion_destino': 'station_id_lock',
                   'bici_sexo': 'sex',
                   'bici_edad': 'age',
                   'bici_direccion': 'address',
                   'bici_pais': 'country',}, inplace=True)
df["day_released"]=""
df["month_released"]=""
df["year_released"]=""
df["hour_released"]=""
df["minute_released"]=""
df["hour_use"]=""
df["minute_use"]=""
df["minute_use_total"]=""


df["day_released"]=pd.to_numeric(df.date_released.str.slice(0, 2), errors="force")
df["month_released"]=pd.to_numeric(df.date_released.str.slice(3, 5), errors="force")
df["year_released"]=pd.to_numeric(df.date_released.str.slice(6, 10), errors="force")
df["hour_released"]=pd.to_numeric(df.date_released.str.slice(11, 13), errors="force")
df["minute_released"]=pd.to_numeric(df.date_released.str.slice(14, 16), errors="force")

df["hour_use"]=pd.to_numeric(df.time_use.str.slice(-8,-6), errors="force")
df["minute_use"]=pd.to_numeric(df.time_use.str.slice(-5,-3), errors="force")

df["minute_use_total"]=df["hour_use"]*60+df["minute_use"]

df.loc[(df.age.isnull()),'age']=30 # just 1 bike

df_normal=df[(df.minute_use_total<=120)]
df_outlier=df[(df.minute_use_total>120)]

print('There are %d bikes with less than 120 minutes in used' % len(df_normal))
print('There are %d bikes with more than 120 minutes in used' % len(df_outlier))
print("Total Database")
print('The longest trip was: %d minutes' % max(df["minute_use_total"]))
print('The shortest trip was: %d minutes' % min(df["minute_use_total"]))
print('The mean trip was: %d minutes' % df["minute_use_total"].mean())
print('The median trip was: %d minutes' % df["minute_use_total"].median())
print('The most frequent trip was: %d minutes' % df["minute_use_total"].mode())
print("")
print("Normal Database")
print('The longest trip was: %d minutes' % max(df_normal["minute_use_total"]))
print('The shortest trip was: %d minutes' % min(df_normal["minute_use_total"]))
print('The mean trip was: %d minutes' % df_normal["minute_use_total"].mean())
print('The median trip was: %d minutes' % df_normal["minute_use_total"].median())
print('The most frequent trip was: %d minutes' % df_normal["minute_use_total"].mode())
print("")
print("Outlier Database")
print('The longest trip was: %d minutes' % max(df_outlier["minute_use_total"]))
print('The shortest trip was: %d minutes' % min(df_outlier["minute_use_total"]))
print('The mean trip was: %d minutes' % df_outlier["minute_use_total"].mean())
print('The median trip was: %d minutes' % df_outlier["minute_use_total"].median())
print('The most frequent trip was: %d minutes' % df_outlier["minute_use_total"].mode())
fig, ax = plt.subplots(figsize=(15,7))
ax = sns.distplot(df_normal.minute_use_total, hist=True, kde=True, 
             bins=int(60), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
ax.set_title("Probability Density Distribution minute_use_total", fontsize=30)

g=df.groupby("station_name_released").agg({'id_user':'count'}).sort_values(by="id_user", ascending=0).head(25)
g=g.plot(kind="bar")
g.set_ylabel("Released bikes", fontsize=10)
g.set_xlabel("Stations", fontsize=10)
g.set_title("Station Distribution (top 25)", fontsize=20)
fig=plt.gcf()
fig.set_size_inches(15,6)
f,ax=plt.subplots(1,2,figsize=(16,5),sharex=True)
df["sex"].value_counts().plot.pie(ax=ax[0], shadow=True)
ax[0].set_title("Sex Distribution", fontsize=20)
ax[0].set_ylabel("")
ax[0].set_xlabel("")
sns.countplot("sex", data=df, ax=ax[1])
ax[1].set_title("Sex Distribution", fontsize=20)
ax[1].set_ylabel("")
ax[1].set_xticklabels(ax[1].get_xticklabels(), ha="right")
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
ax = sns.distplot(df_normal.hour_released, hist=True, kde=False, 
             bins=int(24), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
ax.set_title("Hour Released Distribution", fontsize=30)
ax.set_xlabel("Hour Released", fontsize=10)

fig, ax = plt.subplots(figsize=(15,7))
ax = sns.distplot(df_normal.age, hist=True, kde=False,
             bins=int(82),      
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})
ax.set_title("Age Distribution", fontsize=30)
ax.set_xlabel("Age", fontsize=10)
plt.xlim(16,80)
df.groupby(['year_released','month_released','day_released']).count()['id_user']
fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['year_released','month_released','day_released']).count()['id_user'].plot(ax=ax)
ax.set_title('Bike Released by Day', fontsize=25)
ax.set_xlabel("From 2015 to 2018")
df[(df.year_released==2017)].groupby(['month_released','day_released']).count()['id_user']
fig, ax = plt.subplots(figsize=(15,7))
df[(df.year_released==2017)].groupby(['month_released','day_released']).count()['id_user'].plot(ax=ax)
ax.set_title('Bike Released by Day (2017)', fontsize=25)
ax.set_xlabel("Year2017 (from January to June)")
g=df[(df.country!="Argentina")].groupby("country").agg({'id_user':'count'}).sort_values(by="id_user", ascending=0).head(10)
g=g.plot(kind="bar")
g.set_ylabel("Count Users", fontsize=10)
g.set_xlabel("Country", fontsize=10)
g.set_title("Foreign Users (not argentinian)", fontsize=20)
fig=plt.gcf()
fig.set_size_inches(15,6)



