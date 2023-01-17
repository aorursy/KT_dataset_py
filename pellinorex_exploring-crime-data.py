import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#matplotlib inline

import seaborn as sns

from mpl_toolkits.basemap import Basemap
df = pd.read_csv('../input/crime.csv')
df.head()
df1 = pd.to_datetime(df.Dispatch_Date_Time)

df.Dispatch_Date_Time = df1



df.sort_values(by='Dispatch_Date_Time', inplace=True)

df.index = np.array(range(df.shape[0]))



df['Month'] = df1.dt.month

df['year'] = df1.dt.year

df['day'] = df1.dt.day
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1,figsize=(8,15))



sns.countplot(x='year',data=df, ax=ax1)

ax1.set_ylabel('number of crimes')

ax1.set_title('number of crimes by year')



sns.countplot(x='day',data=df, ax=ax2)

ax2.set_ylabel('number of crimes')

ax2.set_title('number of crimes by day')



sns.countplot(x='Month',data=df, ax=ax3)

ax3.set_ylabel('number of crimes')

ax3.set_title('number of crimes by Month')



sns.countplot(x='Dc_Dist',hue='year',data=df, ax=ax4)

ax1.set_ylabel('number of crimes')

ax1.set_title('number of crimes by year and distriction')



fig.tight_layout()
fig2, ax5 = plt.subplots(figsize=(8,5))

index_order = df['Text_General_Code'].value_counts().index

sns.countplot(x='Text_General_Code',hue='year',data=df, order = index_order,ax=ax5)

ax5.set_ylabel('number of crimes')

ax5.set_title('number of crimes by different cases and year')

ax5.set_xticklabels(labels=index_order,rotation=90)

ax5.legend(bbox_to_anchor=(1, 1),

           bbox_transform=plt.gcf().transFigure)
case_firearm = df.loc[df['Text_General_Code'].isin(

        ['Aggravated Assault Firearm','Robbery Firearm', 'Weapon Violations']

    )]
fig3, (ax6,ax7) = plt.subplots(2,1,figsize=(8,15))



sns.countplot(x='year',data=case_firearm, ax=ax6)

ax6.set_ylabel('number of crimes invovling firearms')

ax6.set_title('number of crimes invovling firearms by year')



sns.countplot(x='Dc_Dist', hue='year',data=case_firearm, ax=ax7)

ax7.set_ylabel('number of crimes invovling firearms by regions')

ax7.set_title('number of crimes invovling firearms by regions and year')

ax7.legend(bbox_to_anchor=(1, 1),

           bbox_transform=plt.gcf().transFigure)
case_firearm_by_hour = case_firearm.groupby('Hour')['Dc_Dist'].count()
fig4, ax8 = plt.subplots(figsize=(8,5))

sns.countplot(x='Hour', data=case_firearm, ax=ax8)
street = lambda x: ' '.join(x['Location_Block'].split(' ')[-2:])

# apply the function to return a new column of last two words in street column

df['Street Name']= df.apply(street, axis =1)

# count apperance times

times =df.groupby(['Street Name'])['Hour'].count()



times = times.to_frame().reset_index()



times = times[times.Hour>10000]



street_crime = df[df['Street Name'].isin(list(times['Street Name']))]
fig5, ax9 = plt.subplots(figsize=(10,10))

m = Basemap(projection='mill', llcrnrlat=df.Lat.min(), urcrnrlat=df.Lat.max(), 

            llcrnrlon=df.Lon.min(), urcrnrlon=df.Lon.max(), resolution='c',epsg=4269, ax=ax9)

x, y = m(tuple(case_firearm.Lon[(case_firearm.Lon.isnull()==False) & (case_firearm.year == 2015)]), \

         tuple(case_firearm.Lat[(case_firearm.Lat.isnull() == False) & (case_firearm.year == 2015)]))



#m.arcgisimage(service="NatGeo_World_Map", xpixels=400)

m.plot(x,y,'ro',markersize=3, alpha=.3, color='red' )



x3, y3 = m(tuple(street_crime.Lon[(street_crime.Lon.isnull()==False) & (street_crime.year == 2015)]), \

         tuple(street_crime.Lat[(street_crime.Lat.isnull() == False) & (street_crime.year == 2015)]))



m.plot(x3,y3,'ro',markersize=2, alpha=.3, color='blue' )



ax9.set_title('Occurance of cases involving firearm')
fig7, ax10 = plt.subplots(figsize=(10,10))

rape_2015 = df[(df['year']== 2015) &(df['Text_General_Code']=='Rape')]



m1 = Basemap(projection='mill',llcrnrlat=df.Lat.min(), urcrnrlat=df.Lat.max(), 

            llcrnrlon=df.Lon.min(), urcrnrlon=df.Lon.max(), resolution='c',epsg=4269,ax=ax10)

x1, y1 = m1(tuple(rape_2015.Lon), tuple(rape_2015.Lat))

#m1.arcgisimage(service="ESRI_StreetMap_World_2D", xpixels=3000)

m1.plot(x1,y1,'ro',markersize=3, alpha=.3,color='red' )



ax10.set_title('Occurance of Rape')
fig8, ax11 = plt.subplots(figsize=(10,10))

crime_night = df.loc[(df['Hour']< 6)].append(df.loc[(df['Hour']>=22)])



crime_night.sort_values(by='Dispatch_Date_Time', inplace=True)



m2 = Basemap(projection='mill', llcrnrlat=df.Lat.min(), urcrnrlat=df.Lat.max(), 

            llcrnrlon=df.Lon.min(), urcrnrlon=df.Lon.max(), resolution='c',epsg=2005, ax=ax11)

x2, y2 = m2(tuple(crime_night[crime_night.year ==2015].Lon), tuple(crime_night[crime_night.year ==2015].Lat))

#m2.arcgisimage(service="ESRI_StreetMap_World_2D", xpixels=3000)

m2.plot(x2,y2,'ro',markersize=1, alpha=.3 ) 



ax11.set_title('Occurance of cases in night')
fig6, ax12 = plt.subplots(figsize=(8,5))



sns.countplot(x='Dc_Dist',data=case_firearm, ax=ax12)

ax12.set_ylabel('number of crimes in night')

ax12.set_title('number of crimes in night by region')