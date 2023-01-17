import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from pandas import Series

%matplotlib inline





from shapely.geometry import shape

import geopandas as gpd

from geopandas import GeoDataFrame

import shapely

import shapely.wkt

from shapely.geometry import Point

from shapely.geometry import Polygon

import math



properati_2013_08 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2013-08-01-properties-sell.csv')

properati_2013_09 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2013-09-01-properties-sell.csv')

properati_2013_10 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2013-10-01-properties-sell.csv')

properati_2013_11 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2013-11-01-properties-sell.csv')

properati_2013_12 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2013-12-01-properties-sell.csv')



properati_2014_01 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-01-01-properties-sell.csv')

properati_2014_02 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-02-01-properties-sell.csv')

properati_2014_03 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-03-01-properties-sell.csv')

properati_2014_04 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-04-01-properties-sell.csv')

properati_2014_05 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-05-01-properties-sell.csv')

properati_2014_06 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-06-01-properties-sell.csv')

properati_2014_07 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-07-01-properties-sell.csv')

properati_2014_08 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-08-01-properties-sell.csv')

properati_2014_09 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-09-01-properties-sell.csv')

properati_2014_10 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-10-01-properties-sell.csv')

properati_2014_11 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-11-01-properties-sell.csv')

properati_2014_12 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2014-12-01-properties-sell.csv')



properati_2015_01 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-01-01-properties-sell.csv')

properati_2015_02 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-02-01-properties-sell.csv')

properati_2015_03 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-03-01-properties-sell.csv')

properati_2015_04 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-04-01-properties-sell.csv')

properati_2015_05 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-05-01-properties-sell.csv')

properati_2015_06 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-06-01-properties-sell.csv')

properati_2015_07 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-07-01-properties-sell.csv')

properati_2015_08 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-08-01-properties-sell.csv')

properati_2015_09 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-09-01-properties-sell.csv')

properati_2015_10 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-10-01-properties-sell.csv')

properati_2015_11 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-11-01-properties-sell.csv')

properati_2015_12 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2015-12-01-properties-sell.csv')


properati_2016_01 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-01-01-properties-sell.csv')

properati_2016_02 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-02-01-properties-sell.csv')

properati_2016_03 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-03-01-properties-sell.csv')

properati_2016_04 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-04-01-properties-sell.csv')

properati_2016_05 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-05-01-properties-sell.csv')

properati_2016_06 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-06-01-properties-sell.csv')

properati_2016_07 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-07-01-properties-sell.csv')

properati_2016_08 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-08-01-properties-sell.csv')

properati_2016_09 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-09-01-properties-sell.csv')

properati_2016_10 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-10-01-properties-sell.csv')

properati_2016_11 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-11-01-properties-sell.csv')

properati_2016_12 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2016-12-01-properties-sell.csv')


properati_2017_01 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-01-01-properties-sell.csv')

properati_2017_02 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-02-01-properties-sell.csv')

properati_2017_03 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-03-01-properties-sell.csv')

properati_2017_04 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-04-01-properties-sell.csv')

properati_2017_05 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-05-01-properties-sell.csv')

properati_2017_06 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-06-01-properties-sell.csv')

properati_2017_07 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-07-01-properties-sell.csv')

properati_2017_08 = pd.read_csv('../tp1/properatiCSVs/properati-AR-2017-08-01-properties-sell.csv')
properati_2013_08.columns.values
len(properati_2013_08) -len(properati_2013_08.loc[properati_2013_08.created_on.str.contains("2013-08"),:]) ==0


properati_2013_08.loc[ properati_2013_08.created_on.str.contains("2013-08")== False,:].tail()


properati_2013_08 = properati_2013_08.loc[ properati_2013_08.created_on.str.contains("2013-08"),:]

properati_2013_09 = properati_2013_09.loc[ properati_2013_09.created_on.str.contains("2013-09"),:]

properati_2013_10 = properati_2013_10.loc[ properati_2013_10.created_on.str.contains("2013-10"),:]

properati_2013_11 = properati_2013_11.loc[ properati_2013_11.created_on.str.contains("2013-11"),:]

properati_2013_12 = properati_2013_12.loc[ properati_2013_12.created_on.str.contains("2013-12"),:]



properati_2014_01 = properati_2014_01.loc[ properati_2014_01.created_on.str.contains("2014-01"),:]

properati_2014_02 = properati_2014_02.loc[ properati_2014_02.created_on.str.contains("2014-02"),:]

properati_2014_03 = properati_2014_03.loc[ properati_2014_03.created_on.str.contains("2014-03"),:]

properati_2014_04 = properati_2014_04.loc[ properati_2014_04.created_on.str.contains("2014-04"),:]

properati_2014_05 = properati_2014_05.loc[ properati_2014_05.created_on.str.contains("2014-05"),:]

properati_2014_06 = properati_2014_06.loc[ properati_2014_06.created_on.str.contains("2014-06"),:]

properati_2014_07 = properati_2014_07.loc[ properati_2014_07.created_on.str.contains("2014-07"),:]

properati_2014_08 = properati_2014_08.loc[ properati_2014_08.created_on.str.contains("2014-08"),:]

properati_2014_09 = properati_2014_09.loc[ properati_2014_09.created_on.str.contains("2014-09"),:]

properati_2014_10 = properati_2014_10.loc[ properati_2014_10.created_on.str.contains("2014-10"),:]

properati_2014_11 = properati_2014_11.loc[ properati_2014_11.created_on.str.contains("2014-11"),:]

properati_2014_12 = properati_2014_12.loc[ properati_2014_12.created_on.str.contains("2014-12"),:]



properati_2015_01 = properati_2015_01.loc[ properati_2015_01.created_on.str.contains("2015-01"),:]

properati_2015_02 = properati_2015_02.loc[ properati_2015_02.created_on.str.contains("2015-02"),:]

properati_2015_03 = properati_2015_03.loc[ properati_2015_03.created_on.str.contains("2015-03"),:]

properati_2015_04 = properati_2015_04.loc[ properati_2015_04.created_on.str.contains("2015-04"),:]

properati_2015_05 = properati_2015_05.loc[ properati_2015_05.created_on.str.contains("2015-05"),:]

properati_2015_06 = properati_2015_06.loc[ properati_2015_06.created_on.str.contains("2015-06"),:]

properati_2015_07 = properati_2015_07.loc[ properati_2015_07.created_on.str.contains("2015-07"),:]

properati_2015_08 = properati_2015_08.loc[ properati_2015_08.created_on.str.contains("2015-08"),:]

properati_2015_09 = properati_2015_09.loc[ properati_2015_09.created_on.str.contains("2015-09"),:]

properati_2015_10 = properati_2015_10.loc[ properati_2015_10.created_on.str.contains("2015-10"),:]

properati_2015_11 = properati_2015_11.loc[ properati_2015_11.created_on.str.contains("2015-11"),:]

properati_2015_12 = properati_2015_12.loc[ properati_2015_12.created_on.str.contains("2015-12"),:]



properati_2016_01 = properati_2016_01.loc[ properati_2016_01.created_on.str.contains("2016-01"),:]

properati_2016_02 = properati_2016_02.loc[ properati_2016_02.created_on.str.contains("2016-02"),:]

properati_2016_03 = properati_2016_03.loc[ properati_2016_03.created_on.str.contains("2016-03"),:]

properati_2016_04 = properati_2016_04.loc[ properati_2016_04.created_on.str.contains("2016-04"),:]

properati_2016_05 = properati_2016_05.loc[ properati_2016_05.created_on.str.contains("2016-05"),:]

properati_2016_06 = properati_2016_06.loc[ properati_2016_06.created_on.str.contains("2016-06"),:]

properati_2016_07 = properati_2016_07.loc[ properati_2016_07.created_on.str.contains("2016-07"),:]

properati_2016_08 = properati_2016_08.loc[ properati_2016_08.created_on.str.contains("2016-08"),:]

properati_2016_09 = properati_2016_09.loc[ properati_2016_09.created_on.str.contains("2016-09"),:]

properati_2016_10 = properati_2016_10.loc[ properati_2016_10.created_on.str.contains("2016-10"),:]

properati_2016_11 = properati_2016_11.loc[ properati_2016_11.created_on.str.contains("2016-11"),:]

properati_2016_12 = properati_2016_12.loc[ properati_2016_12.created_on.str.contains("2016-12"),:]



properati_2017_01 = properati_2017_01.loc[ properati_2017_01.created_on.str.contains("2017-01"),:]

properati_2017_02 = properati_2017_02.loc[ properati_2017_02.created_on.str.contains("2017-02"),:]

properati_2017_03 = properati_2017_03.loc[ properati_2017_03.created_on.str.contains("2017-03"),:]

properati_2017_04 = properati_2017_04.loc[ properati_2017_04.created_on.str.contains("2017-04"),:]

properati_2017_05 = properati_2017_05.loc[ properati_2017_05.created_on.str.contains("2017-05"),:]

properati_2017_06 = properati_2017_06.loc[ properati_2017_06.created_on.str.contains("2017-06"),:]

properati_2017_07 = properati_2017_07.loc[ properati_2017_07.created_on.str.contains("2017-07"),:]

properati_2017_08 = properati_2017_08.loc[ properati_2017_08.created_on.str.contains("2017-08"),:]
interseccion = pd.merge(properati_2014_05,properati_2014_06, how='inner', on=['operation', 'property_type', 'place_name',

       'place_with_parent_names', 'geonames_id', 'lat-lon', 'lat', 'lon',

       'price', 'currency', 'price_aprox_local_currency',

       'price_aprox_usd', 'surface_in_m2', 'price_usd_per_m2', 'floor',

       'rooms', 'expenses', 'properati_url', 'image_thumbnail'])

len(interseccion) == 0


all_properati = pd.concat([properati_2013_08,properati_2013_09,properati_2013_10,properati_2013_11,

                        properati_2013_12,properati_2014_01,properati_2014_02,properati_2014_03,

                        properati_2014_04,properati_2014_05,properati_2014_06,properati_2014_07,

                        properati_2014_08,properati_2014_09,properati_2014_10,properati_2014_11,

                        properati_2014_12,properati_2015_01,properati_2015_02,properati_2015_03,

                        properati_2015_04,properati_2015_05,properati_2015_06,properati_2015_07,

                        properati_2015_08,properati_2015_09,properati_2015_10,properati_2015_11,

                        properati_2015_12,properati_2016_01,properati_2016_02,properati_2016_03,

                        properati_2016_04,properati_2016_05,properati_2016_06,properati_2016_07,

                        properati_2016_08,properati_2016_09,properati_2016_10,properati_2016_11,

                        properati_2016_12,properati_2017_01,properati_2017_02,properati_2017_03,

                        properati_2017_04,properati_2017_05,properati_2017_06,properati_2017_07,

                        properati_2017_08])

#Vista final de cómo quedó el dataframe

all_properati.head(3)
#Analizando los tipos de datos

all_properati.dtypes

#Filtro los que no tienen precio por metro cuadrado

all_properati = all_properati.loc[all_properati.price_usd_per_m2.notnull(),:]
#Filtro solo capital federal y gran buenos aires

capitalFederal = all_properati.loc[all_properati.state_name.str.contains("Capital Federal", na=False),:]

GBA =all_properati.loc[all_properati.state_name.str.contains("G.B.A", na=False),:]

properati = pd.concat([capitalFederal,GBA])
#Convert Date, add Year, Weekday and Hour columns

import calendar

properati['created_on'] = pd.to_datetime(properati['created_on'])



properati[['price_aprox_usd']].describe()
(count, division)= np.histogram(properati.price_usd_per_m2)

bins = []

for i in range(0,len(division)-1):

    bins.append((division[i],division[i+1]))



bins = pd.DataFrame({'bin': bins,

                     'cantidad': count})

bins
properati_filtered = properati[(properati.price_usd_per_m2 < properati.price_usd_per_m2.quantile(.95)) & (properati.price_usd_per_m2 > properati.price_usd_per_m2.quantile(.02))]

sns.set(style="white", palette="muted", color_codes=True)

rs = np.random.RandomState(10)



# Set up the matplotlib figure

f, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.despine(left=True)





g1=sns.distplot(properati[['price_usd_per_m2']],ax=axes[0])

g1.axes.set_title('Precios de propiedades sin filtrar', fontsize=24)



g2 = sns.distplot(properati_filtered[['price_usd_per_m2']], color="purple",ax=axes[1])

g2.axes.set_title('Precios de propiedades filtrada', fontsize=24,  color="purple")



plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()





price_time = properati_filtered.loc[:,['created_on','price_usd_per_m2']]

properati_filtered.groupby('created_on').mean()['price_usd_per_m2'].plot(figsize=(15,4));

plt.title('Inflacion en el area inmobiliaria en funcion del dia',fontsize= 19)

plt.xlabel('Fecha', fontsize=17)

plt.ylabel('Precio USD/m2', fontsize=17)

plt.show()





price_time['yearAndMonth'] = properati_filtered.created_on.map(lambda date: pd.to_datetime(str(date.year)+"-"+str(date.month)+"-01"))

price_time.groupby('yearAndMonth').mean()['price_usd_per_m2'].plot(figsize=(15,4),c='b');

plt.title('Inflacion en el area inmobiliaria en funcion del mes',fontsize= 19)

plt.xlabel('Fecha', fontsize=17)

plt.ylabel('Precio USD/m2', fontsize=17)

plt.show()



price_time['year'] = properati_filtered.created_on.map(lambda date: date.year)

price_time.groupby('year').mean()['price_usd_per_m2'].plot(figsize=(15,4), c='b');

plt.title('Infracion en el area inmobiliaria en funcion del anio',fontsize= 19)

plt.xlabel('Fecha', fontsize=17)

plt.ylabel('Precio USD/m2', fontsize=17)

plt.show()


properati['created_on'].hist(figsize=(16,8));

plt.xlabel('Fechas', fontsize=15);

plt.ylabel('Cantidad de Ventas', fontsize=20)

plt.title('Cantidad de ventas en relacion con el tiempo', fontsize=20);

plt.show();


PBI_trimestres = pd.Series([6898031.78613736,8248262.42985877,8247713.05474478,8806974.38230151], index= ['1ºT','2ºT','3ºT','4ºT'])


pd.options.mode.chained_assignment = None



properati_2016 = properati.loc[(properati['created_on'].dt.year == 2016)]

properati_2016["day"] = properati_2016['created_on'].map(lambda x: x.day)

properati_2016["month"] = properati_2016['created_on'].map(lambda x: x.month)

properati_2016["year"] = properati_2016['created_on'].map(lambda x: x.year)



primer_trim_2016 = properati_2016.loc[properati_2016.month <= 3,:]

monto_primer_trim = primer_trim_2016['price_usd_per_m2'].sum()

mayores_3 = properati_2016['month'] >= 4

menores_7 = properati_2016['month'] <= 6

segundo_trim_2016 = properati_2016[mayores_3 & menores_7]

monto_segundo_trim = segundo_trim_2016['price_usd_per_m2'].sum()

tercer_trim_2016 = properati_2016.loc[(properati_2016.month >=7) & ( properati_2016.month<= 9),:]

monto_tercer_trim = tercer_trim_2016['price_usd_per_m2'].sum()

cuarto_trim_2016 = properati_2016.loc[10 <= properati_2016.month,:]

monto_cuarto_trim = cuarto_trim_2016['price_usd_per_m2'].sum()



#Una vez que se obtiene los montos totales de cada trimestre en el 2016 se hace una Serie con esos valores para luego graficar

montos_trimestres_2016 = pd.Series([monto_primer_trim, monto_segundo_trim, monto_tercer_trim, monto_cuarto_trim], index= ['1ºT','2ºT','3ºT','4ºT'])



PBI_trimestres.plot.line(figsize=(15,10), color='darkorange', fontsize=15);

montos_trimestres_2016.plot.line(figsize=(15,10), color='darkred', fontsize=15);

plt.xlabel('Trimestres', fontsize=18)

plt.ylabel('Valor en dolares', fontsize=20)

plt.title('Correlacion PBI/precio propiedades', fontsize=20);

plt.grid(True)

legend = plt.legend(fontsize=20)

legend.get_texts()[0].set_text('PBI')

legend.get_texts()[1].set_text('Propiedades')



plt.show()
tipos_propiedades_gba = GBA['property_type']

tipos_propiedades_cf = capitalFederal['property_type']

tipos_propiedades_gba.value_counts()


tipos_propiedades_cf.value_counts()


capitalFederal_filtrada = capitalFederal.loc[:,['property_type','place_name']]

capitalFederal_filtrada.dropna(inplace=True)

capitalFederal_filtrada = capitalFederal_filtrada['property_type'].apply(lambda x: Series(x).value_counts()).sum().sort_values(ascending= False )

capitalFederal_filtrada.plot.barh(figsize=(12,5),color='black', fontsize=15);

plt.title('Cantidad de propiedades vendidas en Capital Federal', fontsize=20);

plt.xlabel('Cantidad', fontsize=17);

plt.ylabel('Tipo de propiedad', fontsize=17);

plt.show();
properati_type = properati_filtered.groupby('property_type').mean()['price_usd_per_m2'].sort_values(ascending=False)

graph = properati_type.plot.barh(figsize=(14,5),fontsize=15, color='r', alpha=0.6)

graph.set_title("Precio(usd/m2) promedio segun el tipo de propiedad", fontsize=20)

graph.set_ylabel("Tipo de propiedad", fontsize=18)

graph.set_xlabel("USD/m2", fontsize=18)

plt.show()
properati_gba = GBA.loc[:,['price_usd_per_m2','property_type','created_on']]

properati_caba = capitalFederal.loc[:,['price_usd_per_m2','property_type','created_on']]

properati_gba.dropna(inplace=True)

properati_caba.dropna(inplace=True)
#Variación precio de distintas propiedades a lo largo de los años para el Gran Buenos Aires

fig = plt.figure(figsize=(10,4));



ax = fig.add_axes([0,0,1,1]);

properati_gba.loc[properati_gba['property_type'] == "house"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='r', label="House");

properati_gba.loc[properati_gba['property_type'] == "apartment"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='b', label="Apartment");

properati_gba.loc[properati_gba['property_type'] == "PH"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='g', label="PH");

properati_gba.loc[properati_gba['property_type'] == "store"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='y', label="Store");

plt.title('Precios de diferentes tipos de propiedades a lo largo del tiempo en el GBA', fontsize=23);

plt.xlabel('Fecha', fontsize=16);

plt.ylabel('Precio por metro cuadrado en dolares', fontsize=16);

ax.legend(fontsize=13);

plt.show();
#Variación precio de distintas propiedades a lo largo de los años para Capital Federal

fig = plt.figure(figsize=(10,4));



ax = fig.add_axes([0,0,1,1]);

properati_caba.loc[properati_caba['property_type'] == "house"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='r', label="House");

properati_caba.loc[properati_caba['property_type'] == "apartment"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='b', label="Apartment");

properati_caba.loc[properati_caba['property_type'] == "PH"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='g', label="PH");

properati_caba.loc[properati_caba['property_type'] == "store"].groupby('created_on').mean().loc[:,'price_usd_per_m2'].plot.line(linewidth=2,figsize=(12,8), c='y', label="Store");

plt.title('Precios de diferentes tipos de propiedades a lo largo del tiempo en Capital Federal', fontsize=23);

plt.xlabel('Fecha', fontsize=16);

plt.ylabel('Precio por metro cuadrado en dolares', fontsize=16);

ax.legend(fontsize=13);

plt.show();


fig, ax = plt.subplots(figsize=(16,8))         # Sample figsize in inches

cor = all_properati.corr().abs()

cor.values[[np.arange(6)]*2] = 0

ax = plt.axes()

sns.heatmap(cor,cmap='Purples', alpha=0.5,ax=ax);

ax.set_title('Niveles de correlacion entre los datos', fontsize=20)

plt.show();
#Se filtra las propiedades con más de 6 cuartos

properati_room = properati_filtered.loc[properati_filtered.rooms < 6,['rooms','price_usd_per_m2']]

grouped_rooms = properati_room.groupby('rooms').size().plot(kind='pie', figsize=(5,5),cmap='Accent',fontsize=11)

plt.title('Cantidad de cuartos', fontsize = 18)

plt.ylabel('')

plt.show();


properati_room.groupby('rooms').mean()['price_usd_per_m2'].plot(figsize=(14,4),fontsize=11);

plt.title('Precio en USD/m2 en relacion a la cantidad de cuartos que posee la propiedad', fontsize= 20);

plt.xlabel('Cuartos', fontsize = 14);

plt.ylabel('Precio USD/m2', fontsize=14);

plt.show();

g = sns.boxplot(x='rooms',y='price_usd_per_m2',data=properati_room)

g.set_ylabel('USD/m2',size=18)

g.set_title('Distribucion del precio segun la cantidad de habitaciones',fontsize=18),

g.set_xlabel('cantidad de habitaciones',size=18)
properati_floor = properati_filtered.loc[properati_filtered.floor <10,['floor','price_usd_per_m2']]

properati_floor.groupby('floor').size().plot(kind='pie', figsize=(7,7), fontsize= 13,cmap="Accent")

plt.title('Cantidad de pisos', fontsize=20);

plt.ylabel(' ');

plt.show();
properati_floor.groupby('floor').mean()['price_usd_per_m2'].plot(figsize=(14,4));

plt.title('Precios por metro cuadrado en funcion de la cantidad de pisos', fontsize= 30);

plt.xlabel('Piso', fontsize = 14);

plt.ylabel('Precio USD/m2', fontsize=14);

plt.show();
zone = properati_filtered.groupby('state_name').mean()['price_usd_per_m2'].sort_values(ascending=False)[0:19]

graph2 = zone.sort_values().plot.barh(figsize=(14,5),color='g',fontsize=15, alpha=0.6);

graph2.set_title("Precio(usd/m2) promedio segun la zona", fontsize=20)

graph2.set_ylabel("Zona", fontsize=18)

graph2.set_xlabel("USD/m2", fontsize=18)

plt.show()
boxprops = dict(linestyle='-', linewidth=4, color='k')

medianprops = dict(linestyle='-', linewidth=4, color='k')



g = properati_filtered.loc[:,['state_name','price_usd_per_m2']].boxplot(by='state_name',column='price_usd_per_m2',figsize=(15,5),fontsize=15,boxprops=boxprops,

                medianprops=medianprops)

g.set_title("distribucion del precio segun la zona", fontsize=28)

g.set_xlabel("Zona", fontsize=18)

g.set_ylabel("USD/m2", fontsize=18)
def get_top_places(df, num, cheapest):

    return df.groupby(['place_name'])['price_usd_per_m2'].mean().sort_values(ascending=cheapest).head(num)

    

def plot_top_places(df, num, cheapest,colour,title):

    graph = get_top_places(df, num, cheapest).sort_values().plot.barh(figsize=(15,5),color=colour,fontsize=15)

    graph.set_title(title, fontsize=20)

    graph.set_ylabel("Barrio", fontsize=18)

    graph.set_xlabel("USD/m2", fontsize=18)

    plt.show()

def plot_box(df, size):

    box = df.boxplot(by='place_name',vert=False,column='price_usd_per_m2',figsize=size,fontsize=15,boxprops=boxprops,

                medianprops=medianprops)

    box.set_ylabel("Barrio", fontsize=18)

    box.set_xlabel("USD/m2", fontsize=18)

capitalFederal = properati_filtered.loc[properati_filtered.state_name.str.contains("Capital Federal", na=False),:]

GBA =properati_filtered.loc[properati_filtered.state_name.str.contains("G.B.A", na=False),:]



Zona_Norte =GBA.loc[GBA.state_name.str.contains("Zona Norte", na=False),:]

Zona_Oeste =GBA.loc[GBA.state_name.str.contains("Zona Oeste", na=False),:]

Zona_Sur=GBA.loc[GBA.state_name.str.contains("Zona Sur", na=False),:]

plot_top_places(capitalFederal,5, False,'darkred',"Top 5 de barrios mas caros en Capital Federal")



top5_cheap_CABA = get_top_places(capitalFederal,4, True)

graph4 = top5_cheap_CABA.sort_values().plot.barh(figsize=(15,5),color='bisque',edgecolor='darkred',lineWidth = 1.5,fontsize=15)

graph4.set_title("Top 4 de barrios mas economicos en Capital Federal", fontsize=20)

graph4.set_ylabel("Barrio", fontsize=18)

graph4.set_xlabel("USD/m2", fontsize=18)

plt.show()
plot_box(capitalFederal,(5,50))




barrios_areas = gpd.read_file('../tp1/data_extra/barrios/barrios_badata.shp')

barrios_areas = barrios_areas.loc[:,['BARRIO','PERIMETRO','AREA','geometry']]



barrios_price = capitalFederal.loc[:,['place_name', 'price_usd_per_m2','price_aprox_usd']]

barrios_price.place_name = barrios_price.place_name.str.upper()



precio_m2_promedio = barrios_price.groupby(['place_name'])['price_usd_per_m2'].mean()



barrios_precio = pd.DataFrame({'BARRIO':precio_m2_promedio .index, 'precio_m2_promedio':precio_m2_promedio .values})





barrios = barrios_areas.merge(barrios_precio, on='BARRIO', how='inner')







# plot the shapefile using plot()

figsize=(15,25)

ax = barrios.plot(legend=True, column='precio_m2_promedio', cmap='OrRd', scheme="Quantiles", figsize= (figsize),linewidths=3,edgecolors='k')

plt.title('Promedio de precio/m2 segun el barrio de capital federal', fontsize = 20)
plot_top_places(GBA,5, False,'purple',"Top 5 de barrios mas caros en el Gran Buenos Aires")



plot_top_places(GBA,5, True,'purple',"Top 5 de barrios mas economicos en el Gran Buenos Aires")





plot_box(Zona_Norte,(5,70))
plot_box(Zona_Oeste,(5,50))
plot_box(Zona_Sur,(5,50))
el_Zorzal = GBA.loc[GBA.place_name == 'El Zorzal']

el_Zorzal

barrios = GBA.groupby(['place_name']).size()

barrios_filtrado = barrios[barrios > 8]

barrios_filtrado_df = pd.DataFrame({'place_name':barrios_filtrado.index})



GBA_filtrado = pd.merge(GBA, barrios_filtrado_df,how= 'inner', on ='place_name')

plot_top_places(GBA,5, False,'gray',"Barrios de CABA y GBA ordenados desde el mas caro")

plot_top_places(GBA_filtrado,5, True,'gray',"Barrios de CABA y GBA ordenados desde el mas barato ")

f, axes = plt.subplots(1, 2, figsize=(15, 5))



g1 = properati_filtered.plot.scatter(x='surface_covered_in_m2', y='price_aprox_usd',ax=axes[0])

g1.set_title("Precio en funcion superficie cubierta (sin filtrar)", fontsize=20)

g1.set_ylabel("m2", fontsize=18) 

g1.set_xlabel("USD", fontsize=18)



#filtramos casos aislados de superficie

properati_filtered2 = properati_filtered[(properati_filtered.surface_covered_in_m2 < 

                                      properati_filtered.surface_covered_in_m2.quantile(.95)) 

                                     & (properati_filtered.surface_covered_in_m2 > 

                                        properati_filtered.surface_covered_in_m2.quantile(.02))]



g2 = properati_filtered2.plot.scatter(x='surface_covered_in_m2', y='price_aprox_usd',ax=axes[1]) 

g2.set_title("Precio en funcion superficie cubierta(filtrado)", fontsize=20)

g2.set_ylabel("m2", fontsize=18)

g2.set_xlabel("USD", fontsize=18) 



           

plt.show()
f, axes = plt.subplots(1, 2, figsize=(15, 5))



g1 = capitalFederal.plot.scatter(x='surface_covered_in_m2', y='price_aprox_usd',ax=axes[0])

g1.set_title("Precio en funcion de los m2  en CABA (sin filtrar)", fontsize=20)

g1.set_ylabel("m2", fontsize=18)

g1.set_xlabel("USD", fontsize=18)



#filtramos casos aislados de superficie

capitalFederal2 = capitalFederal[(capitalFederal.surface_covered_in_m2< capitalFederal.surface_covered_in_m2.quantile(.95)) & (capitalFederal.surface_covered_in_m2> capitalFederal.surface_covered_in_m2.quantile(.02))]

capitalFederal2 = capitalFederal2[(capitalFederal2.price_aprox_usd< capitalFederal2.price_aprox_usd.quantile(.90)) & (capitalFederal2.price_aprox_usd> capitalFederal2.price_aprox_usd.quantile(.05))]





g2 = capitalFederal2.plot.scatter(x='surface_covered_in_m2', y='price_aprox_usd',ax=axes[1]) 

g2.set_title("Precio en funcion de los m2 en CABA (filtrado)", fontsize=20)

g2.set_ylabel("m2", fontsize=18)

g2.set_xlabel("USD", fontsize=18) 



           

plt.show()



Puerto_Madero =  capitalFederal2.loc[capitalFederal2.place_name.str.contains("Puerto Madero", na=False),:]

Palermo_Chico =  capitalFederal2.loc[capitalFederal2.place_name.str.contains("Palermo Chico", na=False),:]

Abasto =  capitalFederal2.loc[capitalFederal2.place_name.str.contains("Abasto", na=False),:]

Lugano =  capitalFederal2.loc[capitalFederal2.place_name.str.contains("Lugano", na=False),:]

CABA_4barrios = pd.concat([Puerto_Madero,Palermo_Chico,Abasto,Lugano])

g = sns.lmplot(x='surface_covered_in_m2', y='price_aprox_usd', hue='place_name', col='place_name',data=CABA_4barrios, aspect=.7, x_jitter=2)
gba_filter2 = properati_filtered2.loc[properati_filtered2.state_name.str.contains("G.B.A"),['place_name','price_aprox_usd','surface_covered_in_m2']]

La_lucila =  gba_filter2.loc[gba_filter2.place_name.str.contains("La Lucila", na=False),:]

La_Matanza =  gba_filter2.loc[gba_filter2.place_name.str.contains("La Matanza", na=False),:]

Quilmes =  gba_filter2.loc[gba_filter2.place_name == "Quilmes",:]



GBA_3barrios = pd.concat([La_lucila,La_Matanza,Quilmes ])

g = sns.lmplot(x='surface_covered_in_m2', y='price_aprox_usd', hue='place_name', col='place_name',data=GBA_3barrios, aspect=.9, x_jitter=.1)
def plot_all_country(df):

    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    latitude= -38

    longitude = -60

    m = Basemap(

                ax=axes,

                projection='cyl',

                lon_0=longitude, lat_0=latitude,

                llcrnrlat=latitude - 18.0, urcrnrlat=latitude + 18.0,

                llcrnrlon=longitude - 20.0, urcrnrlon=longitude + 20.0,

                resolution='i'

                )

    m.drawmapboundary(fill_color='#81c0da')

    m.fillcontinents(color='#e3e1a5',lake_color='#81c0da')

    m.drawcoastlines()

    m.drawstates()

    m.drawcountries()



    # Plot the geolocical events as blue dots

    (x, y) = m(df.lon, df.lat )

    m.plot(x, y, 'bo', alpha=0.5, color = 'coral')

    (x, y) = m([longitude], [latitude])

    m.plot(x, y, 'rx', markersize=15.0, color = 'coral')



    plt.title('Ubicacion de propiedades vendidas',fontsize=23)

    plt.show()

    
plot_all_country(properati_filtered)
# Primero vamos a filtrar por latitud y longitud



lat_max = properati_filtered.loc[properati_filtered.lat> -35 ,:]

lat_min = lat_max.loc[lat_max.lat < -34.2]



properati_xy = lat_min.loc[lat_min.lon > -60,:]



plot_all_country(properati_xy)
properati_xy =  properati_xy.loc[properati_xy.price_usd_per_m2 > 2,:]

properati_xy = properati_xy.loc[properati_xy.price_usd_per_m2 < 61900.0,:]



properati_xy['bin'] = pd.qcut(properati_xy.price_usd_per_m2, 10)

bins_count = properati_xy.loc[:,['bin','price_usd_per_m2']].groupby('bin').sum()



graph = bins_count.plot.barh(figsize=(14,5),fontsize=15, alpha=0.6)

graph.set_title("Histograma para precios por metro cuadrado", fontsize=20)

graph.set_ylabel("Bins de USD/m2", fontsize=18)

graph.set_xlabel("cantidad", fontsize=18)

plt.show()
caras =  properati_xy.loc[properati_xy.price_usd_per_m2 > 3500,:]

medias_max =  properati_xy.loc[properati_xy.price_usd_per_m2 < 1900,:]

medias = medias_max.loc[medias_max.price_usd_per_m2 > 1700,:]

baratas =  properati_xy.loc[properati_xy.price_usd_per_m2 < 600,:]
def plot_map_with_dots(df, col,title):

    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    latitude= -35

    longitude = -58

    m = Basemap(ax=axes,projection='cyl',lon_0=longitude, lat_0=latitude,

                llcrnrlat=latitude, urcrnrlat=latitude + 0.8,

                llcrnrlon=longitude - 1.0, urcrnrlon=longitude,

                resolution='i')

    m.drawmapboundary(fill_color='#81c0da')

    m.fillcontinents(color='#e3e1a5',lake_color='#81c0da')

    m.drawcoastlines()

    m.drawstates()

    m.drawcountries()



    # Plot the geolocical events as blue dots

    (x, y) = m(df.lon, df.lat )

    m.plot(x, y, 'bo', alpha=0.5, color = col)

    plt.title(title,fontsize=15)

    plt.show()

plot_map_with_dots(caras,'darkred','Ubicacion de propiedades vendidas a un precio alto')
plot_map_with_dots(medias,'red','Ubicacion de propiedades vendidas a un precio medio')
plot_map_with_dots(baratas,'coral','Ubicacion de propiedades vendidas a un precio bajo')


g =sns.jointplot(x='lon', y='lat', data=properati_xy, kind="kde")

size= (25,15)

plt.rcParams["figure.figsize"] = size

properati_xy.plot.hexbin(x='lon', y='lat', C='price_usd_per_m2', reduce_C_function=np.max,gridsize=150,cmap='winter',fontsize=15)

plt.title('Distribucion geografica de las propiedades segun su precio en Buenos Aires',fontsize=25)

plt.ylabel('Latitud', fontsize=25)

plt.show();
properati_CABA_xy = properati_xy.loc[properati_xy.state_name.str.contains("Capital Federal", na=False),:]
size= (13,13)

plt.rcParams["figure.figsize"] = size

properati_CABA_xy.plot.hexbin(x='lon', y='lat', C='price_usd_per_m2', reduce_C_function=np.max,gridsize=200,cmap='winter')

plt.title('Distribucion geografica de las propiedades seun su precio en Capital Federal',fontsize=21)

plt.ylabel('Latitud', fontsize=22)

plt.show();
#creamos un archivo shape para cada una



#caras

caras_CABA =  caras.loc[caras.state_name.str.contains("Capital Federal", na=False),:]

geometry = [Point(xy) for xy in zip(caras_CABA.lon, caras_CABA.lat)]

caras_CABA_shape = caras_CABA.drop(['lon', 'lat'], axis=1)

crs = {'init': 'epsg:4326'}

geo_caras_CABA = GeoDataFrame(caras_CABA_shape, crs=crs, geometry=geometry)



#medias

medias_CABA =  medias.loc[medias.state_name.str.contains("Capital Federal", na=False),:]

geometry = [Point(xy) for xy in zip(medias_CABA.lon, medias_CABA.lat)]

medias_CABA_shape = medias_CABA.drop(['lon', 'lat'], axis=1)

crs = {'init': 'epsg:4326'}

geo_medias_CABA = GeoDataFrame(medias_CABA_shape, crs=crs, geometry=geometry)



#baratas

baratas_CABA =  baratas.loc[baratas.state_name.str.contains("Capital Federal", na=False),:]

geometry = [Point(xy) for xy in zip(baratas_CABA.lon, baratas_CABA.lat)]

baratas_CABA_shape = baratas_CABA.drop(['lon', 'lat'], axis=1)

crs = {'init': 'epsg:4326'}

geo_baratas_CABA = GeoDataFrame(baratas_CABA_shape, crs=crs, geometry=geometry)
barrios_csv = pd.read_csv('../tp1/data_extra/barrios/barrios.csv', encoding='latin1')

geometry = barrios_csv['WKT'].map(shapely.wkt.loads)

barrios_csv =  barrios_csv.drop('WKT', axis=1)

crs = {'init': 'epsg:4326'}

barrios_geo = gpd.GeoDataFrame(barrios_csv, crs=crs, geometry=geometry)

subte_csv = pd.read_csv('../tp1/data_extra/subte/lineas-de-subte.csv', encoding='latin1')

geometry = subte_csv['WKT'].map(shapely.wkt.loads)

subte_csv =  subte_csv.drop('WKT', axis=1)

crs = {'init': 'epsg:4326'}

subte_geo = gpd.GeoDataFrame(subte_csv, crs=crs, geometry=geometry)
size= (20,15)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



barrios_geo.plot(ax=ax, figsize= size)

geo_caras_CABA.plot(ax=ax,figsize= size,color='darkred')

subte_geo.plot(ax=ax,column='LINEASUB', cmap='Paired')



plt.title('Lineas de subte en capital Federal y propiedades caras', fontsize = 22)

plt.show()
size= (20,15)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



barrios_geo.plot(ax=ax, figsize= size)

geo_medias_CABA.plot(ax=ax,figsize= size,color='r')

subte_geo.plot(ax=ax,column='LINEASUB', cmap='Paired')



plt.title('Lineas de subte en capital Federal y propiedades de valor medio', fontsize = 22)

plt.show()
size= (20,15)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



barrios_geo.plot(ax=ax, figsize= size)

geo_baratas_CABA.plot(ax=ax,figsize= size,color='coral')

subte_geo.plot(ax=ax,column='LINEASUB', cmap='Paired')



plt.title('Lineas de subte en capital Federal y propiedades baratas', fontsize = 22)

plt.show()
def get_distance(lat1,lon1,lat2,lon2):

    return math.sqrt((lat1-lat2)**2 +(lon2-lon1)**2)
def get_min_dist_to_station(lat,lon,estaciones_csv):

    min_dist = 1000

    linea = ""

    estacion = ""

    for index, row in estaciones_csv.iterrows():

        new_distance = get_distance(lat,lon,row['lat'], row['lon'])

        if(new_distance < min_dist):

            min_dist = new_distance

            linea = row['LINEA']

            estacion = row['ESTACION']

    return (min_dist,linea,estacion)
estaciones_subte_csv = pd.read_csv('../tp1/data_extra/subte/estaciones-de-subte.csv', encoding='latin1')

estaciones_subte_csv = estaciones_subte_csv.rename(columns={'Y':'lat'})

estaciones_subte_csv = estaciones_subte_csv.rename(columns={'X':'lon'})



properati_subtes = properati_xy.loc[properati_xy.state_name.str.contains("Capital Federal", na=False), ['lat','lon','price_usd_per_m2'] ]

distancias = []

lineas = []

estaciones =[]

for index, row in properati_subtes.iterrows():

    lat = row['lat']

    lon = row['lon']

    distancia, linea,estacion = get_min_dist_to_station(lat,lon,estaciones_subte_csv)

    distancias.append(distancia)

    lineas.append(linea)

    estaciones.append(estacion)
properati_subtes = properati_xy.loc[properati_xy.state_name.str.contains("Capital Federal", na=False), ['lat','lon','price_usd_per_m2'] ]

properati_subtes['distancia'] = distancias

properati_subtes['linea'] = lineas

properati_subtes['estacion'] = estaciones

properati_subtes.tail()

properati_subtes['bin'] = pd.qcut(properati_subtes.distancia, 10)

graph = properati_subtes.groupby('bin').mean()['price_usd_per_m2'].plot(figsize=(15,5),color='k');

graph.set_title("Precio(usd/m2) promedio segun la distancia a las estaciones de subte", fontsize=20)

graph.set_xlabel("Distancia a una estacion de subte", fontsize=18)

graph.set_ylabel("USD/m2", fontsize=18)

plt.show()
properati_linea = properati_subtes.groupby('linea').mean()['price_usd_per_m2'].sort_values(ascending=False)

graph = properati_linea.plot(kind ='bar',figsize=(14,5),fontsize=15, color=['g','gold','r','b','turquoise','purple'])

graph.set_title("Precio(usd/m2) promedio segun la cervania a las lineas de subte", fontsize=20)

graph.set_xlabel("Linea de subte mas cercana", fontsize=18)

graph.set_ylabel("USD/m2", fontsize=18)

plt.show()
def plot_subte_linea(letra,color):

    linea = properati_subtes.loc[properati_subtes.linea == letra,:]

    linea['bin'] = pd.qcut(linea.distancia, 10)

    graph = linea.groupby('bin').mean()['price_usd_per_m2'].plot(figsize=(15,5), color=color);

    graph.set_title("Precio(usd/m2) promedio segun la distancia a las estaciones de subte "+letra, fontsize=20)

    graph.set_xlabel("Distancia a una estacion de subte linea "+letra, fontsize=18)

    graph.set_ylabel("USD/m2", fontsize=18)

    plt.show()
def plot_subte_estaciones(letra,color):

    properati_linea = properati_subtes.loc[properati_subtes.linea == letra,:].groupby('estacion').mean()['price_usd_per_m2'].sort_values(ascending=False)

    graph = properati_linea.plot.barh(figsize=(10,5),fontsize=15, color=color)

    graph.set_title("Precio(usd/m2) promedio segun la cercania a las  estaciones de subte "+letra, fontsize=20)

    graph.set_xlabel("promedio USD/m2 de las propiedades cercanas", fontsize=18)

    graph.set_ylabel("Estaciones de subte de la linea "+letra, fontsize=18)

    plt.show()

plot_subte_linea('D','g')

plot_subte_estaciones('D','g')
plot_subte_linea('H','gold')

plot_subte_estaciones('H','gold')
plot_subte_linea('B','r')

plot_subte_estaciones('B','r')
plot_subte_linea('C','b')

plot_subte_estaciones('C','b')
plot_subte_linea('A',"turquoise")

plot_subte_estaciones('A','turquoise')
plot_subte_linea('E',"purple")



plot_subte_estaciones('E',"purple")
estaciones_list = properati_subtes.groupby('estacion').mean()['price_usd_per_m2']

properati_estaciones = pd.DataFrame({'ESTACION':estaciones_list.index, 'precio_m2_promedio':estaciones_list.values})



estaciones_ubicacion = estaciones_subte_csv.loc[:,['ESTACION','lat','lon']]

properati_estaciones = properati_estaciones.merge(estaciones_ubicacion).sort_values(by='precio_m2_promedio' )



#creo lista de tamaños de puntos 

point_size = []

for n in range(len(properati_estaciones)):

    point_size.append(2 +(n**2)*2)





sizes = pd.DataFrame({'size':point_size} , index=properati_estaciones.index)



properati_estaciones['size'] = sizes

size= (50,40)



plt.rcParams["figure.figsize"] = size



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



barrios_geo.plot(ax=ax, figsize= size, color='bisque',linewidths=1,edgecolors='gray')

plt.scatter(properati_estaciones.lon,properati_estaciones.lat, s=properati_estaciones['size'],alpha=0.7,linewidths=3,edgecolors='k')

plt.title('Estaciones de subte en la Capital Federal', fontsize=60)

plt.show()
def get_distance_to_center(lat,lon):

    lat_obelisco = -34.603075

    lon_obelisco = -58.381653

    return get_distance(lat,lon,lat_obelisco,lon_obelisco)



def get_column_distance_to_center(df):

    distances = []

    for index, row in df.iterrows():

        lat = row['lat']

        lon = row['lon']

        distances.append(get_distance_to_center(lat,lon))

    return distances
properati_center = properati_xy.loc[:, ['lat','lon','expenses','price_usd_per_m2'] ]

properati_center['distance_center'] = get_column_distance_to_center(properati_center)

properati_center['bin'] = pd.qcut(properati_center.distance_center, 10)

graph = properati_center.groupby('bin').mean()['price_usd_per_m2'].plot.area(color='k',figsize=(15,5));

graph.set_title("Precio(usd/m2) promedio segun la distancia al centro", fontsize=20)

graph.set_xlabel("Distancia al centro", fontsize=18)

graph.set_ylabel("USD/m2", fontsize=18)

plt.show()
esp_verdes_csv = pd.read_csv('../tp1/data_extra/espacio-verde-publico.csv', encoding='latin1')

geometry = esp_verdes_csv['WKT'].map(shapely.wkt.loads)

esp_verdes_csv =  esp_verdes_csv.drop('WKT', axis=1)

crs = {'init': 'epsg:4326'}

esp_verdes_geo = gpd.GeoDataFrame(esp_verdes_csv, crs=crs, geometry=geometry)
size= (15,10)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



barrios_geo.plot(ax=ax, figsize= size, color='bisque')

geo_caras_CABA.plot(ax=ax,figsize= size, color='darkred',alpha=0.5)

geo_medias_CABA.plot(ax=ax,figsize= size, color='red',alpha=0.4)

geo_baratas_CABA.plot(ax=ax,figsize= size)

esp_verdes_geo.plot(ax=ax,color='g')



plt.title('Espacios verdes en Capital Federal junto a las propiedades', fontsize = 22, alpha=0.7)

plt.xlabel('Longitud', fontsize=20)

plt.ylabel('Latitud', fontsize=20)

plt.show()
estaciones_tren_csv = pd.read_csv('../tp1/data_extra/estaciones-de-ferrocarril.csv', encoding='latin1',delimiter=";")

estaciones_tren_csv = estaciones_tren_csv.rename(columns={'LNG':'lon'})

estaciones_tren_csv = estaciones_tren_csv.rename(columns={'LAT':'lat'})

estaciones_tren_csv = estaciones_tren_csv.rename(columns={'NOMBRE':'ESTACION'})

estaciones_tren_csv.tail()
properati_trenes = properati_xy.loc[:,['lat','lon','price_usd_per_m2'] ]

distancias = []

lineas = []

estaciones =[]

for index, row in properati_trenes.iterrows():

    lat = row['lat']

    lon = row['lon']

    distancia, linea,estacion = get_min_dist_to_station(lat,lon,estaciones_tren_csv)

    distancias.append(distancia)

    lineas.append(linea)

    estaciones.append(estacion)
properati_trenes['distancia'] = distancias

properati_trenes['linea'] = lineas

properati_trenes['estacion'] = estaciones

properati_trenes.tail()
properati_trenes['bin'] = pd.qcut(properati_trenes.distancia, 10)

graph = properati_trenes.groupby('bin').mean()['price_usd_per_m2'].plot(figsize=(15,5),color='k')

graph.set_title("Precio(usd/m2) promedio segun la distancia a las estaciones de tren", fontsize=20)

graph.set_xlabel("Distancia a una estacion de tren", fontsize=18)

graph.set_ylabel("USD/m2", fontsize=18)

plt.show()
properati_lineas_tren = properati_trenes.groupby('linea').mean()['price_usd_per_m2'].sort_values()

graph = properati_lineas_tren.plot.barh(figsize=(14,5),fontsize=15, color='turquoise')

graph.set_title("Precio(usd/m2) promedio segun la cercania a las lineas de ferrocarril", fontsize=20)

graph.set_xlabel("USD/m2 de las propiedades cercanas", fontsize=18)

graph.set_ylabel("Linea de tren", fontsize=18)

plt.show()
def plot_linea(linea):

    linea_tren = properati_trenes.loc[properati_trenes.linea.str.contains(linea),:].groupby('estacion').mean()['price_usd_per_m2'].sort_values()

    graph = linea_tren.plot.barh(figsize=(14,5),fontsize=15)

    graph.set_title("Precio(usd/m2) promedio de las propiedades cercanas a las estaciones de la linea de ferrrocarril: "+linea, fontsize=20)

    graph.set_xlabel("USD/m2 de las propiedades cercanas", fontsize=18)

    graph.set_ylabel("Linea de tren", fontsize=18)

    plt.show()
plot_linea('BELGRANO NORTE')
belgrano_norte = estaciones_tren_csv.loc[estaciones_tren_csv.LINEA.str.contains('BELGRANO NORTE'),:]

belgrano_norte 
plot_linea('MITRE')
plot_linea('SAN MARTIN')
plot_linea('TRANVIA')
plot_linea('SARMIENTO')
plot_linea('ROCA')
plot_linea('URQUIZA')
plot_linea('BELGRANO SUR')
estaciones_tren_list = properati_trenes.groupby('estacion').mean()['price_usd_per_m2']

properati_estaciones_tren = pd.DataFrame({'ESTACION':estaciones_tren_list.index, 'precio_m2_promedio':estaciones_tren_list.values})

estaciones_tren_ubicacion = estaciones_tren_csv.loc[:,['ESTACION','lat','lon']]

estaciones_tren_ubicacion.tail()

properati_estaciones_tren = properati_estaciones_tren.merge(estaciones_tren_ubicacion).sort_values(by='precio_m2_promedio' )

#creo lista de tamaños de puntos 

#creo lista de tamaños de puntos 

point_size = []

for n in range(len(properati_estaciones)):

    point_size.append(2 +(n**2)*2)

    

sizes = pd.DataFrame({'size':point_size} , index=properati_estaciones.index)



properati_estaciones_tren['size'] = sizes


size= (50,40)



plt.rcParams["figure.figsize"] = size



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



barrios_geo.plot(ax=ax, figsize= (10,5), color='bisque',linewidths=1,edgecolors='gray')

plt.scatter(properati_estaciones_tren.lon,properati_estaciones_tren.lat, s=properati_estaciones_tren['size'],alpha=0.7,linewidths=3,edgecolors='k')

plt.title('Estaciones de tren en Capital FederaL', fontsize=60)

plt.show()
monumentos_csv = pd.read_csv('../tp1/data_extra/monumentos.csv', encoding='latin1',delimiter=";")

monumentos_csv = monumentos_csv.rename(columns={'LONGITUD':'lon'})

monumentos_csv = monumentos_csv.rename(columns={'LATITUD':'lat'})

monumentos_csv = monumentos_csv.loc[(monumentos_csv.lat.notnull()) & (monumentos_csv.lon.notnull()),['OBJETO_OBRA','lat','lon']]







geometry = [Point(xy) for xy in zip(monumentos_csv.lon, monumentos_csv.lat)]

monumentos_shape = monumentos_csv.drop(['lon', 'lat'], axis=1)

crs = {'init': 'epsg:4326'}

geo_monumentos = GeoDataFrame(monumentos_shape, crs=crs, geometry=geometry)

size= (20,15)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



barrios_geo.plot(ax=ax, figsize= size, color='gray',linewidths=1,edgecolors='w')



plt.scatter(caras_CABA.lon,caras_CABA.lat,color='darkred',marker='x')

plt.scatter(medias_CABA.lon,medias_CABA.lat,color='red',marker='x')

plt.scatter(baratas_CABA.lon,baratas_CABA.lat,color='bisque',marker='x')

plt.title('Monumentos en capital federal', fontsize = 10)



plt.scatter(monumentos_csv.lon,monumentos_csv.lat, s=60,linewidths=3,edgecolors='cyan')

legend = plt.legend(fontsize=20)

legend.get_texts()[0].set_text('Propiedades costosas')

legend.get_texts()[1].set_text('Propiedades de valor medio')

legend.get_texts()[2].set_text('Propiedades eonomicas')

legend.get_texts()[3].set_text('Monumentos')

plt.show()
distritos_economicos_csv = pd.read_csv('../tp1/data_extra/distritos-economicos.csv', encoding='latin1')

geometry = distritos_economicos_csv['WKT'].map(shapely.wkt.loads)

distritos_economicos_csv =  distritos_economicos_csv.drop('WKT', axis=1)

crs = {'init': 'epsg:4326'}

distritos_economicos_geo = gpd.GeoDataFrame(distritos_economicos_csv, crs=crs, geometry=geometry)



size= (10,10)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')

barrios_geo.plot(ax=ax, figsize= size, color='gray',linewidths=1,edgecolors='w')

distritos_economicos_geo.plot(ax=ax, figsize= size)

plt.title('Distritos economicos', fontsize = 18)

plt.show()
barrios_populares = pd.read_csv('../tp1/data_extra/barriospopulares.csv')

figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

latitude= -35

longitude = -58

m = Basemap(ax=axes,projection='cyl',lon_0=longitude, lat_0=latitude,

            llcrnrlat=latitude, urcrnrlat=latitude + 0.8,

            llcrnrlon=longitude - 1.0, urcrnrlon=longitude,

            resolution='i')

m.drawmapboundary(fill_color='#81c0da')

m.fillcontinents(color='#e3e1a5',lake_color='#81c0da')

m.drawcoastlines()

m.drawstates()

m.drawcountries()

coordenadas = barrios_populares['geojson']

for i in range(len(coordenadas)):

    df = pd.read_json(coordenadas[i])

    coords = df['coordinates']

    lon = coords[0][0][0][0]

    lat = coords[0][0][0][1]

    (x,y) = m(lon, lat)

    m.plot(x, y, 'bo', alpha=0.5, color = 'coral')

plt.title('Asentamientos precarios en Gran Buenos Aires',fontsize=25)

plt.show()


cf_bpop = barrios_populares[barrios_populares["provincia_nombre"] == "CABA"]

ba_bpop = barrios_populares[barrios_populares["provincia_nombre"] == "Buenos Aires"]

cf_bpop.groupby(cf_bpop["localidad_comuna_nombre"]).count()


#Se procede a agarrar la comuna 4 que es la que tiene más "barrios populares"

com4 = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 4"]



#Separación los barrios populares por comuna

com1bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 1"]

com2bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 2"]

com4bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 4"]

com7bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 7"]

com8bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 8"]

com9bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 9"]

com14bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 14"]

com15bp = cf_bpop[cf_bpop["localidad_comuna_nombre"] == "Comuna 15"]
#separamos las ventas por comunas, para ver mejor en el gráfico

#ver que esten bien los nombres en capitalfederal2

propcom1 = capitalFederal2[(capitalFederal2["place_name"] == "Retiro") | (capitalFederal2["place_name"] == "San Telmo") |

                          (capitalFederal2["place_name"] == "Montserrat") | (capitalFederal2["place_name"] == "Constitución") |

                           (capitalFederal2["place_name"] == "San Nicolás") | (capitalFederal2["place_name"] == "Puerto Madero")] 

propcom2 = capitalFederal2[(capitalFederal2["place_name"] == "Recoleta")]

propcom4 = capitalFederal2[(capitalFederal2["place_name"] == "Barracas") | (capitalFederal2["place_name"] == "Boca") |

                          (capitalFederal2["place_name"] == "Parque Patricios") |

                          (capitalFederal2["place_name"] == "Pompeya")]

propcom7 = capitalFederal2[(capitalFederal2["place_name"] == "Flores") | (capitalFederal2["place_name"] == "Parque Chacabuco")]

propcom8 = capitalFederal2[(capitalFederal2["place_name"] == "Villa Soldati") | 

                           (capitalFederal2["place_name"] == "Villa Riachuelo") |

                          (capitalFederal2["place_name"] == "Villa Lugano")]

propcom9 = capitalFederal2[(capitalFederal2["place_name"] == "Liniers") | 

                           (capitalFederal2["place_name"] == "Parque Avellaneda") |

                          (capitalFederal2["place_name"] == "Mataderos")]

propcom14 = capitalFederal2[(capitalFederal2["place_name"] == "Palermo")]

propcom15 = capitalFederal2[(capitalFederal2["place_name"] == "Chacarita") |

                            (capitalFederal2["place_name"] == "Villa Crespo")]
com1bp.columns.values
import geojson



capitalFederal2bis = capitalFederal2.loc[capitalFederal2['lat']<-32.5,:]

capitalFederal2bis = capitalFederal2bis.loc[capitalFederal2bis['lon']>-60,:]

capitalFederal2bis = capitalFederal2bis.loc[capitalFederal2bis['lon']<-58.3,:]



#propiedades en venta

caba_points = capitalFederal2bis[['lon', 'lat']].apply(lambda row: Point(row["lon"], row["lat"]), axis=1)

geo_cabaprop = gpd.GeoDataFrame({"geometry": caba_points, "price_aprox_usd": capitalFederal2bis['price_aprox_usd']})



#com4geo: "barrios populares"

cf_bpop.geojson

cf_bpop['geometry'] = cf_bpop['geojson'].apply(lambda row: shape(geojson.loads(row))) 

cf_bpopgeo = gpd.GeoDataFrame({'geometry': cf_bpop.geometry, 'barrio_nombre':cf_bpop['barrio_nombre']}, crs=crs)

#comunacsvtogeo: contorno de comuna

#comunasshp = gpd.read_file('comunas/comunas.shp')

#comunasshp.crs = crs

#comunasshp.plot()

#comunas = comunas_csv[comunas_csv["COMUNAS"]==14]

geometry = comunas_csv['WKT'].map(shapely.wkt.loads)

comunas =  comunas_csv.drop('WKT', axis=1)

crs = {'init': 'epsg:4326'}

comunasgeo = gpd.GeoDataFrame(comunas, crs=crs, geometry=geometry)



size= (20,15)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



comunasgeo.plot(ax=ax, figsize= size, color='grey')

geo_cabaprop.plot(ax=ax,figsize= size,c=geo_cabaprop.price_aprox_usd,

                  legend=True, column='price_aprox_usd')

cf_bpopgeo.plot(ax=ax, color='Blue')



plt.title('Propiedades en venta y "Barrios populares" en comuna 4', fontsize = 10)

plt.show()
import geojson

comunas_csv = pd.read_csv('../tp1/data_extra/comunas.csv', encoding='latin1')



#propiedades en venta

propcom1bis = propcom1.loc[propcom1['lat']<-34.4,:]

propcom1bis = propcom1bis.loc[propcom1bis['lat']>-34.64,:]

propcom1bis = propcom1bis.loc[propcom1bis['lon']>-58.44,:]

comprop_points = propcom1bis[['lon', 'lat']].apply(lambda row: Point(row["lon"], row["lat"]), axis=1)

geo_comprop = gpd.GeoDataFrame({"geometry": comprop_points, "price_aprox_usd": propcom1bis['price_aprox_usd']})

crs = {'init': 'epsg:4326'}

#com4geo: "barrios populares"

com1bp['geometry'] = com1bp['geojson'].apply(lambda row: shape(geojson.loads(row))) 

comgeo = gpd.GeoDataFrame({'geometry': com1bp.geometry, 'barrio_nombre':com1bp['barrio_nombre']}, crs=crs)

#comunacsvtogeo: contorno de comuna

comunas = comunas_csv[comunas_csv["COMUNAS"]==1]

geometry = comunas['WKT'].map(shapely.wkt.loads)

comunas =  comunas.drop('WKT', axis=1)



comunageo = gpd.GeoDataFrame(comunas, crs=crs, geometry=geometry)



size= (20,15)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



comunageo.plot(ax=ax, figsize= size, color='grey')

geo_comprop.plot(ax=ax,figsize= size,c=geo_comprop.price_aprox_usd, legend=True, column='price_aprox_usd')

comgeo.plot(ax=ax, color='Blue')



plt.title('Propiedades en venta y "Barrios populares" en comuna 1', fontsize = 10)

plt.show()
#comuna 4

#propiedades en venta

com4prop_points = propcom4[['lon', 'lat']].apply(lambda row: Point(row["lon"], row["lat"]), axis=1)

geo_com4prop = gpd.GeoDataFrame({"geometry": com4prop_points, "price_aprox_usd": propcom4['price_aprox_usd']})



#com4geo: "barrios populares"

com4['geometry'] = com4['geojson'].apply(lambda row: shape(geojson.loads(row))) 

com4geo = gpd.GeoDataFrame({'geometry': com4.geometry, 'barrio_nombre':com4['barrio_nombre']}, crs=crs)

#comunacsvtogeo: contorno de comuna

comuna = comunas_csv[comunas_csv["COMUNAS"]==4]

geometry = comuna['WKT'].map(shapely.wkt.loads)

comuna =  comuna.drop('WKT', axis=1)

crs = {'init': 'epsg:4326'}

comunascsvtogeo = gpd.GeoDataFrame(comuna, crs=crs, geometry=geometry)



size= (20,15)



fig, ax = plt.subplots(figsize= size)

ax.set_aspect('equal')



comunascsvtogeo.plot(ax=ax, figsize= size, color='grey')

geo_com4prop.plot(ax=ax,figsize= size,c=geo_com4prop.price_aprox_usd,

                  legend=True, column='price_aprox_usd')

com4geo.plot(ax=ax, color='Blue')



plt.title('Propiedades en venta y "Barrios populares" en comuna 4', fontsize = 10)

plt.show()