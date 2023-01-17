import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd # geodata processing à la pandas

from pathlib import Path # python3 pathlib

import re # regular expressions

import folium # to make map with leaflet

import matplotlib.pyplot as plt # to display graphs and plots

import seaborn as sns

import collections # more collection types



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#         print(os.path.join(dirname, filename))
dirpath = Path('../input/donneesgaz2008a2017/')

dataframe_collection = {}

pathlist = dirpath.glob('**/*.csv')



for path in pathlist:

    year = re.findall('_(\d+).csv',str(path))[0]

    dataframe_collection[year] = pd.read_csv(path, sep=';', encoding='latin1', decimal=',', dtype={'CODE':'str'})
for year,df in list(dataframe_collection.items()):

    if df.duplicated().any():

        print("Des lignes en doublons dans le jeu de données de " + year)

        display(df[df.duplicated(keep=False)])

    if not df[df.duplicated(subset=['CODE','OPERATEUR'],keep=False)].empty:

        print("Il existe des doublons sur le couple (code,opérateur) dans le jeu de données de " + year)

        display(df[df.duplicated(subset=['CODE','OPERATEUR'],keep=False)])
for year,df in list(dataframe_collection.items()):

    if df.isnull().values.any():

        print("Des valeurs manquantes dans le jeu de données de " + year)

        display(df[df.isnull().any(axis=1)])
for year,df in list(dataframe_collection.items()):

    commune = df[df.TYPE=="Commune"]

    if(not commune[commune.CODE.str.len()!=5].empty):

        print("Erreurs sur les codes Commune dans le jeu de données de " + year)

        print(commune[commune.CODE.str.len()!=5])



    iris = df[df.TYPE=="IRIS"]

    if(not iris[iris.CODE.str.len()!=9].empty):

        print("Erreurs sur les codes IRIS dans le jeu de données de " + year)

        print(iris[iris.CODE.str.len()!=9])



    region = df[df.TYPE=="Region"]

    if(not region[region.CODE.str.len()!=2].empty):

        print("Erreurs sur les codes Region dans le jeu de données de " + year)

        print(region[region.CODE.str.len()!=2])



    epci = df[df.TYPE=="InterCom"]

    if(not epci[epci.CODE.str.len()!=9].empty):

        print("Erreurs sur les codes InterCom dans le jeu de données de " + year)

        print(epci[epci.CODE.str.len()!=9])



# POUR LES COMMUNES HORS EPCI ?



    if(not epci[epci.CODE=="000000000"].empty):

        print("Code InterCom 000000000 dans le jeu de données de " + year)



    if(not epci[epci.CODE=="ZZZZZZZZZ"].empty):

        print("Code InterCom ZZZZZZZZZ dans le jeu de données de " + year)
df_insee_2017 = pd.read_excel('../input/iris-insee-2017/reference_IRIS_geo2017.xls', skiprows=5)

data_gaz = pd.read_csv('../input/donneesgaz2008a2017/donnees_gaz_2017.csv', sep=';', encoding='latin1', decimal=',', dtype={'CODE':'str'})



# On ne prend pas en compte les codes IRIS se terminant en 9999

df_gaz = data_gaz[(data_gaz.TYPE=='IRIS') & (~(data_gaz.CODE.str.endswith('9999')))].copy()



# On merge les deux fichiers

merge_df = df_gaz.merge(df_insee_2017, left_on='CODE', right_on='CODE_IRIS', how='left', indicator=True)



# On liste les codes IRIS de la donnée elec non apparié avec la donnée INSEE

merge_df= merge_df[merge_df['_merge'] == 'left_only']

print(str(merge_df['CODE'].nunique()) + " codes IRIS uniques non appariés")
df_insee_2016 = pd.read_excel('../input/iris-insee-2016/reference_IRIS_geo2016.xls', skiprows=5)



# On merge les deux fichiers

merge_df = df_gaz.merge(df_insee_2016, left_on='CODE', right_on='CODE_IRIS', how='left', indicator=True)



# On liste les codes IRIS de la donnée elec non apparié avec la donnée INSEE

merge_df= merge_df[merge_df['_merge'] == 'left_only']

display(merge_df.head())

print(str(merge_df['CODE'].nunique()) + " codes IRIS uniques non appariés")
od = collections.OrderedDict(sorted(dataframe_collection.items()))

for year, df in list(od.items()):

    print(year)

    print("#####")

    iris = df[(df.TYPE=='IRIS') & (~(df.CODE.str.endswith('9999')))]

    print(str(iris.CODE.nunique()) + " codes IRIS uniques hors *9999 dans le jeu de données de " + year)

    communes = df[(df.TYPE=='Commune')]

    print(str(communes.CODE.nunique()) + " codes Communes uniques dans le jeu de données de " + year)

    epci = df[(df.TYPE=='InterCom') & (df.CODE.str!="000000000") & (df.CODE.str!="ZZZZZZZZZ")]

    print(str(epci.CODE.nunique()) + " codes InterCom uniques hors 000000000 et ZZZZZZZZZ dans le jeu de données de " + year)

    regions = df[(df.TYPE=='Region')]

    print(str(regions.CODE.nunique()) + " code Region uniques dans le jeu de données de " + year)

    print("\n")
df_gaz.reset_index()

df_gaz['conso_totale'] = df_gaz.CONSOA + df_gaz.CONSOI + df_gaz.CONSOT + df_gaz.CONSOR + df_gaz.CONSONA

df_gaz = df_gaz.rename(columns={'CODE':'CODE_IRIS'})

print( str(df_gaz['CODE_IRIS'].nunique()) + " codes IRIS renseignés")
bx = sns.boxplot(df_gaz['conso_totale'])
geodata = gpd.read_file('../input/irisgeojson/iris.geojson')

iris_with_data = geodata.merge(df_gaz, on='CODE_IRIS')

ax = iris_with_data.plot(column='conso_totale', scheme='QUANTILES',k=20)

ax.set_xlim(-5.225,9.55)

ax.set_ylim(41.333, 51.2)

plt.show()