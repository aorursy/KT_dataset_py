import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

import seaborn as sns

print(check_output(["ls", "../input"]).decode("utf8"))
airres = pd.read_csv('../input/air_reserve.csv')

airstore = pd.read_csv('../input/air_store_info.csv')

hpgres = pd.read_csv('../input/hpg_reserve.csv')

hpgstore = pd.read_csv('../input/hpg_store_info.csv')

airvisit = pd.read_csv('../input/air_visit_data.csv')

airvisit.tail()
airres.head()
air = pd.merge(airres,airstore,on='air_store_id')

hpg = pd.merge(hpgres,hpgstore,on='hpg_store_id')

rel = pd.read_csv('../input/store_id_relation.csv')

airrel = pd.merge(air,rel,how='left',on='air_store_id')

hpgrel = pd.merge(hpg,rel,how='left',on='hpg_store_id')

full = pd.merge(airrel,hpgrel,how='outer')

print("there are ",len(air)," restaurants with AIR and ",len(hpg)," with HPG.",len(rel),' have both.')
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

f,ax = plt.subplots(1,1,figsize=(15,16))

m = Basemap(projection='aeqd',width=2000000,height=2000000, lat_0=37.5, lon_0=138.2)



m.drawcoastlines()

m.fillcontinents(color='lightgray',lake_color='aqua',zorder=1)

m.scatter(hpg.longitude.values,hpg.latitude.values,marker='o',color='blue',zorder=999,latlon=True)

m.scatter(air.longitude.values,air.latitude.values,marker='x',color='green',zorder=999,latlon=True)

ax.legend()

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0).fit(full[['longitude','latitude']])

full['cluster'] = kmeans.predict(full[['longitude','latitude']])
cx = [c[0] for c in kmeans.cluster_centers_]

cy = [c[1] for c in kmeans.cluster_centers_]

cm = plt.get_cmap('gist_rainbow')

colors = [cm(2.*i/10) for i in range(10)]

colored = [colors[k] for k in full['cluster']]

f,axa = plt.subplots(1,1,figsize=(15,16))

m.drawcoastlines()

m.fillcontinents(color='lightgray',lake_color='aqua',zorder=1)

m.scatter(full.longitude.values,full.latitude.values,color=colored,s=20,alpha=1,zorder=999,latlon=True)

m.scatter(cx,cy,color='Black',s=50,alpha=1,latlon=True,zorder=9999)

plt.setp(axa.get_yticklabels(), visible=True)

plt.annotate('Fukuoka', xy=(0.04, 0.32), xycoords='axes fraction',fontsize=20)

plt.annotate('Shikoku', xy=(0.25, 0.25), xycoords='axes fraction',fontsize=20)

plt.annotate('Hiroshima', xy=(0.2, 0.36), xycoords='axes fraction',fontsize=20)

plt.annotate('Osaka', xy=(0.40, 0.30), xycoords='axes fraction',fontsize=20)



plt.annotate('Tokyo', xy=(0.60, 0.4), xycoords='axes fraction',fontsize=20)

plt.annotate('Shizoku', xy=(0.50, 0.32), xycoords='axes fraction',fontsize=20)



for i in range(len(cx)):

    xpt,ypt = m(cx[i],cy[i])

    plt.annotate(i, (xpt+500,ypt+500),zorder=99999,fontsize=16)

plt.show()



f,axa = plt.subplots(1,2,figsize=(15,6))

hist_clust = full.groupby(['cluster'],as_index=False).count()

sns.barplot(x=hist_clust.cluster,y=hist_clust.air_store_id,ax=axa[0])

sns.barplot(x=hist_clust.cluster,y=hist_clust.hpg_store_id,ax=axa[1])

plt.show()
f,ax = plt.subplots(1,1,figsize=(15,6))

airhist = air.groupby(['air_store_id'],as_index=False).count()

sns.distplot(airhist.visit_datetime)

hpghist = hpg.groupby(['hpg_store_id'],as_index=False).count()

sns.distplot(hpghist.visit_datetime)

plt.show()

#sns.barplot(x= airhist.air_store_id,y = airhist.visit_datetime)
air_genre = full.loc[full.air_genre_name.isnull()==False].groupby(['cluster','air_genre_name'],as_index=False).count()

hpg_genre = full.loc[full.hpg_genre_name.isnull()==False].groupby(['cluster','hpg_genre_name'],as_index=False).count()



genres = air.air_genre_name.unique()



#i = 0

f,axa= plt.subplots(2,1,figsize=(15,36))

hm = []

for i in range(10):

    genres_count = [ air_genre.loc[air_genre.cluster==i].loc[air_genre.air_genre_name==name]['air_store_id'].values[0] if name in air_genre.loc[air_genre.cluster==i].air_genre_name.values else 0 for name in genres] 

    hm.append(genres_count)

hm = pd.DataFrame(hm,columns=genres,)

sns.heatmap(hm.transpose(),cmap="YlGnBu",ax=axa[0])



genres = hpg.hpg_genre_name.unique()

hm = []

for i in range(10):

    genres_count = [ hpg_genre.loc[hpg_genre.cluster==i].loc[hpg_genre.hpg_genre_name==name]['hpg_store_id'].values[0] if name in hpg_genre.loc[hpg_genre.cluster==i].hpg_genre_name.values else 0 for name in genres] 

    hm.append(genres_count)

hm = pd.DataFrame(hm,columns=genres,)

sns.heatmap(hm.transpose(),cmap="YlGnBu",ax=axa[1])





plt.show()
dates = pd.read_csv('../input/date_info.csv')

dates.loc[dates.holiday_flg==1].loc[(dates.day_of_week !='Saturday')].loc[dates.day_of_week !='Sunday']

vdt = pd.to_datetime(full.visit_datetime)

rdt = pd.to_datetime(full.reserve_datetime)

full['vd']=vdt.dt.date

full['vt']=vdt.dt.time

full['rd']=rdt.dt.date

full['rt']=rdt.dt.time



dts = pd.to_datetime(dates.calendar_date)

days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']

dates['calendar_date'] = pd.to_datetime(dates['calendar_date']).dt.date

dates['dy'] = dts.dt.dayofyear

dates['dw'] = [days.index(dw) for dw in dates.day_of_week]

dates.head()



nf = pd.merge(full,dates[['calendar_date','holiday_flg']],how='left',left_on='vd',right_on='calendar_date')

nf = nf.rename(index = str, columns = {'holiday_flg':'visit_holiday'})

nf = nf.drop(['calendar_date'],axis=1)



nf = pd.merge(nf,dates[['calendar_date','holiday_flg']],how = 'left', left_on='rd',right_on='calendar_date')

nf = nf.rename(index = str, columns = {'holiday_flg':'reservation_holiday'})

nf = nf.drop(['calendar_date'],axis=1)



nf['vd'] = pd.to_datetime(nf['vd']).dt.dayofyear

nf['rd'] = pd.to_datetime(nf['rd']).dt.dayofyear

nf.head()

deltatime = vdt - rdt

days = deltatime.dt.days



days.describe()
f,axa = plt.subplots(1,1,figsize=(15,6))

sns.distplot(days)

plt.xlim(0,40)

axa.set_title('Days between Reservation and Visit')

plt.show()
f,ax = plt.subplots(1,1, figsize=(15,6))

vholidayhist= nf[nf['visit_holiday']==1].groupby(['vd'],as_index=False).count()

sns.barplot(x = vholidayhist.vd,y=vholidayhist.visit_datetime)

ax.set_title('Visits in Japanese Holidays')

plt.show()
f,ax = plt.subplots(1,1, figsize=(15,6))

vholidayhist= nf[nf['visit_holiday']==0].groupby(['vd'],as_index=False).count()

sns.barplot(x = vholidayhist.vd,y=vholidayhist.visit_datetime)

ax.set_title('Visits in Other Days')

plt.show()