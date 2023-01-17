import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from scipy.stats import zscore,stats

from sklearn.model_selection import train_test_split
df1 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2009-json_corrigido.json", lines=True)

df2 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2010-json_corrigido.json", lines=True)

df3 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2011-json_corrigido.json", lines=True)

df4 = pd.read_json("../input/nytaxi/data-sample_data-nyctaxi-trips-2012-json_corrigido.json", lines=True)
df_v = pd.read_csv("../input/nytaxi/data-vendor_lookup-csv.csv")

df_p = pd.read_csv("../input/nytaxi/data-payment_lookup-csv.csv", skiprows = 1)
df_all=pd.concat([df1,df2,df3,df4])
df_all.info()
df_all.head(-5)
df_p
df_all = pd.merge(df_all, df_p, on='payment_type')
df_v
df_v.columns=['vendor_id','vendor_name','vendor_address','vendor_city','vendor_state','vendor_zip','vendor_country','vendor_contact','vendor_current']
df_v
df_all = pd.merge(df_all, df_v, on='vendor_id')
df_all.head(-5)
df_all["dropoff_datetime"]=df_all["dropoff_datetime"].astype("datetime64")
df_all.describe()
avd = df_all[['trip_distance']].where(df_all.passenger_count<=2).mean().iloc[0]
print("\n\nThe average distance traveled by trips with a maximum of 2 passengers is:\n{}".format(avd))
df_all.vendor_id.unique()
df_all['total_amount'].groupby(df_all.vendor_id).sum().sort_values()[::-1]
df_all['total_amount'].groupby(df_all.vendor_name).sum().sort_values()[::-1][:3]
vdf = df_all['total_amount'].groupby(df_all.vendor_id).sum().sort_values().reset_index()
pd.merge(vdf[['vendor_id','total_amount']], df_v[['vendor_id','vendor_name']], on='vendor_id')[::-1][:3]
dfv_xy=pd.merge(vdf[['vendor_id','total_amount']], df_v[['vendor_id','vendor_name']], on='vendor_id')[::-1][:3].iloc[:,[2,1]]
dfv_xy
dfv_xy.plot.bar(x='vendor_name', y='total_amount')
df_all.payment_type.unique()
df_all.payment_lookup.unique()
df_all.dropoff_datetime
df_all["dropoff_datetime"]
df_all.dropoff_datetime.groupby(df_all["dropoff_datetime"].dt.month).count().plot(kind="bar",title ='Rides per month for all payment methods' )

plt.xlabel("Month Numbers")

plt.ylabel("Number Of Rides")
df_all.dropoff_datetime.where(df_all.payment_lookup=='Cash').groupby(df_all["dropoff_datetime"].dt.month).count().plot(kind="bar",title ='Rides per month for Cash payment methods')

plt.xlabel("Month Numbers")

plt.ylabel("Number Of Rides")
df_all.dropoff_datetime.where(df_all.payment_lookup=='Cash').groupby(by=[(df_all["dropoff_datetime"].dt.year),(df_all["dropoff_datetime"].dt.month)]).count().plot(figsize=(15, 4),kind="bar",title ='Rides per month for each year, for Cash payment methods')

plt.xlabel("Months")

plt.ylabel("Number Of Rides")
df_all.tip_amount.unique()
df_all.tip_amount.isnull().sum()
df_all.tip_amount.where(df_all.tip_amount==0).count()
df_all.tip_amount.where(df_all.tip_amount!=0).count()
df_all.tip_amount.where((df_all.tip_amount!=0) & (df_all["dropoff_datetime"].dt.year==2012)).count()
df_all.tip_amount.where((df_all.tip_amount!=0) & (df_all["dropoff_datetime"].dt.year==2012)& (df_all["dropoff_datetime"].dt.month>=(8))).count()
months_of_the_year=(df_all["dropoff_datetime"].dt.month).where((df_all["dropoff_datetime"].dt.year==2012)&((df_all["dropoff_datetime"].dt.month)!=np.nan)).unique()
last3thMonth=sorted(months_of_the_year)[::-1][:3][-1]
DaysOfLastThreeMonthsOnCondition=df_all.tip_amount.groupby(df_all["dropoff_datetime"].dt.date.where((df_all.tip_amount!=0) & (df_all["dropoff_datetime"].dt.year==2012)& (df_all["dropoff_datetime"].dt.month>=(last3thMonth)))).count()
DaysOfLastThreeMonthsOnCondition
DaysOfLastThreeMonthsOnCondition.plot(figsize=(16, 4), title="The number of tips each day for the last 3 months of 2012")

plt.xticks()

plt.xlabel("Dates")

plt.ylabel("Tip Amount")



plt.subplots_adjust(bottom=0.15)

plt.show()
days=sorted(df_all["dropoff_datetime"].dt.date.loc[(df_all["dropoff_datetime"].dt.year==2012) & (df_all["dropoff_datetime"].dt.month>=8)].unique())
DaysOfLastThreeMonthsOnCondition.plot(figsize=(16, 4), title= "The number of tips each day for the last 3 months of 2012")

plt.xticks(days, rotation='vertical')

plt.xlabel("Dates")

plt.ylabel("Tip Amount")

plt.show()
df_all[['trip_distance']].where((df_all["dropoff_datetime"].dt.weekday==5)).mean().iloc[0]
df_all[['trip_distance']].where((df_all["dropoff_datetime"].dt.weekday==6)).mean().iloc[0]
df_all[['trip_distance']].where((df_all["dropoff_datetime"].dt.weekday==5)|(df_all["dropoff_datetime"].dt.weekday==6)).mean().iloc[0]
summer = range(172, 264)

fall = range(264, 355)

spring = range(80, 172)





def season(x):

    if x in summer:

       return 'Summer'



    if x in fall:

       return 'Fall'



    if x in spring:

       return 'Spring'



    else :

       return 'Winter'
bins = [0, 91, 183, 275, 366]

labels=['Winter', 'Spring', 'Summer', 'Fall']

doy = df_all["dropoff_datetime"].dt.dayofyear

df_all['SEASONN'] = pd.cut(doy + 11 - 366*(doy > 355), bins=bins, labels=labels)
df_all['SEASONN']
plt.xticks([i * 1 for i in range(0, 4)])

df_all.dropoff_datetime.groupby(by=[(df_all['SEASONN'])]).count().plot(kind="bar",  color=['gray', 'green', 'green', 'gray'], title="Total amount of rides grouped by seasons")

plt.xlabel("Seasons")

plt.ylabel("Rides")
plt.xticks([i * 1 for i in range(0, 16)])

df_all.dropoff_datetime.groupby(by=[(df_all["dropoff_datetime"].dt.year),(df_all['SEASONN'])]).count().plot(figsize=(15, 4),title="Total amount of rides grouped by seasons of each year")

plt.xticks(rotation=30)

plt.xlabel("Seasons by year")

plt.ylabel("Rides")
plt.xticks([i * 1 for i in range(0, 16)])

df_all.dropoff_datetime.groupby(by=[(df_all["dropoff_datetime"].dt.year),(df_all['SEASONN'])]).count().plot(figsize=(15, 4), kind="bar",  color=['gray', 'green', 'green', 'gray'],title="Total amount of rides grouped by seasons of each year")

plt.xticks(rotation=30)

plt.xlabel("Seasons by year")

plt.ylabel("Rides")
df2.columns
BBox = ((df2.pickup_longitude.min(), df2.pickup_longitude.max(), df2.pickup_latitude.min(), df2.pickup_latitude.max()))
BBox
fig, ax = plt.subplots(figsize = (8,7))

ax.scatter(df2.pickup_longitude, df2.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)



ax.set_xlim(BBox[0],BBox[1])

ax.set_ylim(BBox[2],BBox[3])
locv = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
df_loc=df2[locv]
z_scores = stats.zscore(df_loc)



abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3.5).all(axis=1)

new_df = df_loc[filtered_entries]
BBox2 = ((new_df.pickup_longitude.min(), new_df.pickup_longitude.max(), new_df.pickup_latitude.min(), new_df.pickup_latitude.max()))
fig, ax = plt.subplots(figsize = (8,7))

ax.scatter(new_df.pickup_longitude, new_df.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)



ax.set_xlim(BBox2[0],BBox2[1])

ax.set_ylim(BBox2[2],BBox2[3])
z_scores = stats.zscore(new_df)



abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3.5).all(axis=1)

new_df2 = new_df[filtered_entries]
BBox3 = ((new_df2.pickup_longitude.min(), new_df2.pickup_longitude.max(), new_df2.pickup_latitude.min(), new_df2.pickup_latitude.max()))
BBox3
fig, ax = plt.subplots(figsize = (11,11))

ax.scatter(new_df2.pickup_longitude, new_df2.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)



ax.set_xlim(BBox3[0],BBox3[1])

ax.set_ylim(BBox3[2],BBox3[3])
nymap = plt.imread("../input/nytaxi/MapNY.jpg")
fig, ax = plt.subplots(figsize = (17,17))

ax.scatter(new_df2.pickup_longitude, new_df2.pickup_latitude, zorder=1, alpha= 1.0, c='b', s=10, label="pickup")

ax.scatter(new_df2.dropoff_longitude, new_df2.dropoff_latitude, zorder=1, alpha= 0.99, c='r', s=5, label="dropoff")

ax.set_title('Pickup & Dropoff locations in 2010')

ax.set_xlim(BBox3[0],BBox3[1])

ax.set_ylim(BBox3[2],BBox3[3])



plt.legend(loc='upper left',fontsize='large')





ax.imshow(nymap, zorder=0, extent = BBox3, aspect= 'equal')
fig, ax = plt.subplots(ncols=2, figsize = (19,19))



ax[0].scatter(new_df2.pickup_longitude, new_df2.pickup_latitude, zorder=1, alpha= 1.0, c='b', s=10, label="pickup")

ax[1].scatter(new_df2.dropoff_longitude, new_df2.dropoff_latitude, zorder=1, alpha= 0.99, c='r', s=5, label="dropoff")



ax[0].set_title('Pickup locations in 2010')

ax[1].set_title('Dropoff locations in 2010')



ax[0].set_xlim(BBox3[0],BBox3[1])

ax[0].set_ylim(BBox3[2],BBox3[3])



ax[0].legend(loc='upper left',fontsize='large')

ax[1].legend(loc='upper left',fontsize='large')



ax[0].imshow(nymap, zorder=0, extent = BBox3, aspect= 'equal')



ax[1].set_xlim(BBox3[0],BBox3[1])

ax[1].set_ylim(BBox3[2],BBox3[3])





ax[1].imshow(nymap, zorder=0, extent = BBox3, aspect= 'equal')

loca = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','fare_amount','tolls_amount']
df_loc_all=df_all[loca]
z_scores = stats.zscore(df_loc_all)



abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3.5).all(axis=1)

new_dfa = df_loc_all[filtered_entries]



z_scores = stats.zscore(new_dfa)



abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3.5).all(axis=1)

new_dfa2 = new_dfa[filtered_entries]
fig, ax = plt.subplots(figsize = (11,11))

ax.scatter(new_dfa2.pickup_longitude, new_dfa2.pickup_latitude, zorder=1, alpha= 0.2, c='b', s=10)

ax.scatter(new_dfa2.dropoff_longitude, new_dfa2.dropoff_latitude, zorder=1, alpha= 0.2, c='b', s=10)
new_dfa2
new_dfa2['fare_and_tolls_amount']=new_dfa2.fare_amount+new_dfa2.tolls_amount
new_dfa2['longtitude_distance']=abs(new_dfa2.dropoff_longitude-new_dfa2.pickup_longitude)
new_dfa2['latitude_distance']=abs(new_dfa2.dropoff_latitude-new_dfa2.pickup_latitude)
import math

new_dfa2['hypotenuse']=np.sqrt((new_dfa2['longtitude_distance']**2)+(new_dfa2['latitude_distance']**2))
X = new_dfa2[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','longtitude_distance','latitude_distance','hypotenuse']].values

y = new_dfa2['fare_and_tolls_amount'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.linear_model import LinearRegression



r = LinearRegression()

r.fit(X_train, y_train)
print("Test set R^2 score is: {:.2f}".format(r.score(X_test, y_test)))
from sklearn.model_selection import cross_val_score



cross_val_score(r, X_test, y_test, cv=3).mean()
def predictor(pickup_longitudeX, pickup_latitudeX, dropoff_longitudeX, dropoff_latitudeX):

    lodi=abs(pickup_longitudeX-dropoff_longitudeX)

    ladi=abs(pickup_latitudeX-dropoff_latitudeX)

    hypo=math.sqrt((lodi**2)+(ladi**2))

    

    ar=np.array([[pickup_longitudeX, pickup_latitudeX, dropoff_longitudeX, dropoff_latitudeX, lodi, ladi, hypo]])

    return print("Estimated fare is: ${:.2f}".format(r.predict(ar)[0]))
#predictor(pickup_longitudeX, pickup_latitudeX, dropoff_longitudeX, dropoff_latitudeX):
predictor(-74, 41,-74, 41)
predictor(-74, 41,-74, 41.2)
predictor(-73.948288,40.774511,-73.997466,40.718039)
during_day = df_all.dropoff_datetime.where((df_all["dropoff_datetime"].dt.hour>18)|(df_all["dropoff_datetime"].dt.hour<6)).dropna()
during_night = df_all.dropoff_datetime.where((df_all["dropoff_datetime"].dt.hour<=18)|(df_all["dropoff_datetime"].dt.hour>=6)).dropna()
during_day.groupby(df_all["dropoff_datetime"].dt.dayofyear).count().plot(kind="line",title ='Rides in during day and during night by day of the year', label="During nights")

during_night.groupby(df_all["dropoff_datetime"].dt.dayofyear).count().plot(kind="line", label="During days")

plt.xlabel("Day of the year")

plt.ylabel("Number of rides")

plt.legend(loc='lower center',fontsize='large')
