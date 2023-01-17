import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings('ignore')
#Information about CSV file



properati = pd.read_csv('properati.csv')





print("Shape of the dataframe:",properati.shape,"\n\n")

print("Types of the columns are: ", properati.info(),"\n\n")

print("Null values in columns are: \n",properati.isnull().sum(),"\n\n")

properati.head(1)

#types checking

properati.dtypes
#Price < at median

cpt = 0

median = properati['price_aprox_usd'].median()



for value in properati['price_aprox_usd']:

    if(value <= median): cpt+=1



print("Median inferior houses number: ",cpt)
#buildings type repartition by state

fig = plt.figure(figsize=(8, 6))

fig.clf()

ax = fig.gca()



properati['state_name'].value_counts().plot(ax = ax, kind='bar')

plt.show()
#Fill null values for each columns that will be used



properati["price"] = properati["price"].fillna(value=0)

properati["price_aprox_usd"] = properati["price_aprox_usd"].fillna(value=0)

properati["surface_total_in_m2"] = properati["surface_total_in_m2"].fillna(value=0)

properati["price_usd_per_m2"] = properati["price_usd_per_m2"].fillna(value=0)

properati["surface_covered_in_m2"] = properati["surface_covered_in_m2"].fillna(value=0)

properati["lat"] = properati["lat"].fillna(value=0)

properati["lon"] = properati["lon"].fillna(value=0)



print("Null values in columns are: \n",properati.isnull().sum())
# make a function to plot survival against passenger attribute



#Afficher prix par m2 en fonction de surface m2 (price_aprox_usd, surface_total_in_m2)

#Afficher evolatuion du prix en m2 en fonction de l'année de création (price_per_m2, created_on)



%matplotlib inline





def properati_hist(dataframe, columns):

    for col in columns:

        fig = plt.figure(figsize=(6, 4))

        fig.clf()

        ax = fig.gca()

        ax.set_xlabel(col)

        ax.set_ylabel('Density of ' + col)

        

        print("Median of",col,"is:",dataframe[col].median())

        x = dataframe[col]

        bins = np.linspace(1,dataframe[col].median()+1000, 100)

        x.hist(bins = bins, ax = ax)

    return 'Done'



hist_cols = ["price_aprox_usd", "surface_total_in_m2", "price_per_m2"]

properati_hist(properati, hist_cols)

#Diagramme cammember avec property type



cpt1 = 0

cpt2 = 0

cpt3 = 0

cpt4 = 0



for value in properati["property_type"]:

    if(value == "apartment"):

        cpt1+=1

    if(value == "house"): 

        cpt2+=1

    if(value == "store"): 

        cpt3+=1

    if(value == "PH"): 

        cpt4+=1

    



data = [cpt1, cpt2, cpt3, cpt4]

name = ['Apartment', 'House', 'Store', 'PH']



plt.figure(figsize=(6,6))

plt.title('Distribtion of building type\n', fontsize=20)

explode=(0.15, 0, 0, 0)

plt.pie(data, explode=explode, labels=name, autopct='%1.1f%%', startangle=90, shadow=True,pctdistance=0.87)

plt.axis('equal')

plt.show()
#buildings type repartition by state

fig = plt.figure(figsize=(8, 6))

fig.clf()

ax = fig.gca()



properati['rooms'].value_counts().plot(ax = ax, kind='bar')

plt.show()
#Evolution of the price per m2 in time





import matplotlib.pyplot as plt





fig = plt.figure(figsize=(8, 6))

fig.clf()

ax = fig.gca()



lims = (min(properati['price_per_m2']), 200000) 





properati['created_on'] = pd.to_datetime(properati['created_on'], format='%Y-%m-%d')

properati.plot( x = 'created_on', y = 'price_per_m2',ax = ax, ylim=lims)



plt.xlabel("Date")

plt.ylabel("Price per m2")

plt.title("Evolution of the price per m2 in time\n")



plt.show()
import folium





locations = properati[['lat', 'lon']]

locationlist = locations.values.tolist()

len(locationlist)





map = folium.Map(location=[-34.6371, -58.3672], zoom_start=10)



for point in range(0, 200):

    folium.Marker(locationlist[point], popup=properati['title'][point]).add_to(map)

map