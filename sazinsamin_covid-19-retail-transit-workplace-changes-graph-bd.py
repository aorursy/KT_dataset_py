import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('../input/GMR.csv',parse_dates=True)
df.head(5)
df2=df.copy()
df3=df2.fillna('x',inplace=True)
df3=df2.copy()
df3.head(3)
bd=df3[df3['country_region']=='Bangladesh']
tai=df3[df3['country_region']=='Taiwan']
uk=df3[df3['country_region']=='United Kingdom']
us=df3[df3['country_region']=='United States']
ita=df3[df3['country_region']=='Italy']
fra=df3[df3['country_region']=='France']
ger=df3[df3['country_region']=='Germany']
uk2=uk[uk['sub_region_1']=='x']
us2=us[us['sub_region_1']=='x']
ita2=ita[ita['sub_region_1']=='x']
fra2=fra[fra['sub_region_1']=='x']
ger2=ger[ger['sub_region_1']=='x']
tai.shape
plt.figure(figsize=(30,12))
plt.title('retail and recreation percent change from baseline')
plt.plot(bd['date'],bd['retail_and_recreation_percent_change_from_baseline'])
plt.plot(tai['date'],tai['retail_and_recreation_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['retail_and_recreation_percent_change_from_baseline'])
plt.plot(us2['date'],us2['retail_and_recreation_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['retail_and_recreation_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('retail_and_recreation_percent_change_from_baseline.png')
plt.figure(figsize=(30,12))
plt.title('residential percent change from baseline')
plt.plot(bd['date'],bd['residential_percent_change_from_baseline'])
plt.plot(tai['date'],tai['residential_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['residential_percent_change_from_baseline'])
plt.plot(us2['date'],us2['residential_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['residential_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('residential_percent_change_from_baseline.png')
plt.figure(figsize=(30,12))
plt.title('workplaces percent change from baseline')
plt.plot(bd['date'],bd['workplaces_percent_change_from_baseline'])
plt.plot(tai['date'],tai['workplaces_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['workplaces_percent_change_from_baseline'])
plt.plot(us2['date'],us2['workplaces_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['workplaces_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('workplaces_percent_change_from_baseline.png')
plt.figure(figsize=(30,12))
plt.title('grocery and pharmacy percent change from baseline')
plt.plot(bd['date'],bd['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(tai['date'],tai['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(us2['date'],us2['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['grocery_and_pharmacy_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('grocery_and_pharmacy_percent_change_from_baseline.png')
plt.figure(figsize=(30,12))
plt.title('parks percent change from baseline')
plt.plot(bd['date'],bd['parks_percent_change_from_baseline'])
plt.plot(tai['date'],tai['parks_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['parks_percent_change_from_baseline'])
plt.plot(us2['date'],us2['parks_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['parks_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('parks_percent_change_from_baseline.png')
plt.figure(figsize=(30,12))
plt.title('transit stations percent change from baseline')
plt.plot(bd['date'],bd['transit_stations_percent_change_from_baseline'])
plt.plot(tai['date'],tai['transit_stations_percent_change_from_baseline'])
plt.plot(uk2['date'],uk2['transit_stations_percent_change_from_baseline'])
plt.plot(us2['date'],us2['transit_stations_percent_change_from_baseline'])
plt.plot(ita2['date'],ita2['transit_stations_percent_change_from_baseline'])
plt.legend(['Banglades','Taiwan','United Kindom','United State','Italy'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.savefig('transit_stations_percent_change_from_baseline.png')
ny=us[us['sub_region_1']=='New York']
ny=ny[ny['sub_region_2']=='x']
ny.shape
ny.head(3)
plt.figure(figsize=(30,12))
plt.title('retail_and_recreation_percent_change_from_baseline')
plt.plot(bd['date'],bd['retail_and_recreation_percent_change_from_baseline'])
plt.plot(ny['date'],ny['retail_and_recreation_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.figure(figsize=(30,12))
plt.title('residential_percent_change_from_baseline')
plt.plot(bd['date'],bd['residential_percent_change_from_baseline'])
plt.plot(ny['date'],ny['residential_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.figure(figsize=(30,12))
plt.title('workplaces_percent_change_from_baseline')
plt.plot(bd['date'],bd['workplaces_percent_change_from_baseline'])
plt.plot(ny['date'],ny['workplaces_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.figure(figsize=(30,12))
plt.title('transit_stations_percent_change_from_baseline')
plt.plot(bd['date'],bd['transit_stations_percent_change_from_baseline'])
plt.plot(ny['date'],ny['transit_stations_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.figure(figsize=(30,12))
plt.title('parks_percent_change_from_baseline')
plt.plot(bd['date'],bd['parks_percent_change_from_baseline'])
plt.plot(ny['date'],ny['parks_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.figure(figsize=(30,12))
plt.title('grocery_and_pharmacy_percent_change_from_baseline')
plt.plot(bd['date'],bd['grocery_and_pharmacy_percent_change_from_baseline'])
plt.plot(ny['date'],ny['grocery_and_pharmacy_percent_change_from_baseline'])
plt.legend(['Banglades','New York'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)