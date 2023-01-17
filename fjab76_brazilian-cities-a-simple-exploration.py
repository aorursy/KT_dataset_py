# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/BRAZIL_CITIES.csv", sep=";", decimal=",")
df.head()
df.info()
df.fillna(0., inplace=True)
# Amount of states plus Federal District

df["STATE"].value_counts().shape
# States and cities per state

df["STATE"].value_counts()
# remove zero values

mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 

mask3 = df['CAPITAL'] ==1

 

# use the scatter function

plt.figure(figsize=(10,10))

plt.title("Cities Latitude and Longitude")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(df[mask1&mask2&mask3]["LONG"], df[mask1&mask2&mask3]["LAT"], s=20, alpha=1, label='Capital city')

plt.scatter(df[mask1&mask2&~mask3]["LONG"], df[mask1&mask2&~mask3]["LAT"], s=1, alpha=1, label='Other')

plt.legend()

plt.show()
# remove zero values

mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 

mask3 = df['CAPITAL'] ==1

 

# use the scatter function

plt.figure(figsize=(10,10))

plt.title("Population per Latitude and Longitude")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

scale_factor = 20000

plt.scatter(df[mask1&mask2&mask3]["LONG"], df[mask1&mask2&mask3]["LAT"], s=df[mask1&mask2&mask3]["ESTIMATED_POP"]/scale_factor, alpha=1, label='Capital city')

plt.scatter(df[mask1&mask2&~mask3]["LONG"], df[mask1&mask2&~mask3]["LAT"], s=df[mask1&mask2&~mask3]["ESTIMATED_POP"]/scale_factor, alpha=1, label='Other')

plt.legend()

plt.show()
# total population (2018)

print(f"total population (2018): {df['ESTIMATED_POP'].sum():10.0f}")



# % growth between 2010 and 2018

avg_growth = (df['ESTIMATED_POP'].sum()-df['IBGE_RES_POP'].sum())/df['IBGE_RES_POP'].sum()*100

print(f"% population growth between 2010 and 2018: {avg_growth:2.2f}")
# population by state (2018)

pop_by_state = df[['STATE','ESTIMATED_POP']].groupby(by="STATE").sum().sort_values(by="ESTIMATED_POP", ascending=False)

plt.figure(figsize=(15,10))

plt.bar(pop_by_state.index, pop_by_state['ESTIMATED_POP'])

plt.title("Population by state (2018)")

plt.show()
# Fastest growing states

fastest_growing_states = df[['STATE','IBGE_RES_POP','ESTIMATED_POP']].groupby(by="STATE").sum()

fastest_growing_states['%GROWTH'] = (fastest_growing_states['ESTIMATED_POP']-fastest_growing_states['IBGE_RES_POP'])/fastest_growing_states['IBGE_RES_POP']*100

fgs = fastest_growing_states.sort_values(by="%GROWTH", ascending=False)

plt.figure(figsize=(15,10))

plt.bar(fgs.index, fgs['%GROWTH'], label='% growth')

plt.plot(fgs.index, [avg_growth]*fgs.index.shape[0], color='red', label='% avg growth')

plt.legend()

plt.title("Fastest growing states")

plt.show()
# Fastest growing capital cities

fastest_growing_capitals = df[df['CAPITAL']==1][['CITY','STATE','IBGE_RES_POP','ESTIMATED_POP']]

fastest_growing_capitals['%GROWTH'] = (fastest_growing_capitals['ESTIMATED_POP']-fastest_growing_capitals['IBGE_RES_POP'])/fastest_growing_capitals['IBGE_RES_POP']*100

fgc = fastest_growing_capitals.sort_values(by="%GROWTH", ascending=False)

plt.figure(figsize=(40,20))

plt.bar(fgc['CITY'], fgc['%GROWTH'], label='% growth')

plt.plot(fgc['CITY'], [avg_growth]*fgc.shape[0], color='red', label='% avg growth')

plt.legend()

plt.title("Fastest growing capital cities")

plt.show()
# total foreign population

print(f"total foreign population: {df['IBGE_RES_POP_ESTR'].sum():10.0f}")



# % of foreign population

print(f"% of foreign population {(df['IBGE_RES_POP_ESTR'].sum()/df['IBGE_RES_POP'].sum()*100):10.2f}")
# remove zero values

mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 

mask3 = df["CAPITAL"] ==1 

 

# use the scatter function

plt.figure(figsize=(10,10))

plt.title("Foreign population (2010)")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

scale_factor = 100

plt.scatter(df[mask1&mask2&mask3]["LONG"], df[mask1&mask2&mask3]["LAT"], s=df[mask1&mask2&mask3]["IBGE_RES_POP_ESTR"]/scale_factor, alpha=1, label='Capital city')

plt.scatter(df[mask1&mask2&~mask3]["LONG"], df[mask1&mask2&~mask3]["LAT"], s=df[mask1&mask2&~mask3]["IBGE_RES_POP_ESTR"]/scale_factor, alpha=1, label='Other')

plt.legend()

plt.show()
# remove zero values

mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 

mask3 = df["STATE"] =='PR' 

 

# use the scatter function

plt.figure(figsize=(10,10))

plt.title("Parana population (2018)")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

scale_factor = 3000

plt.scatter(df[mask1&mask2&mask3]["LONG"], df[mask1&mask2&mask3]["LAT"], s=df[mask1&mask2&mask3]["ESTIMATED_POP"]/scale_factor, alpha=1)

plt.show()
df[['CITY','ESTIMATED_POP']][df["STATE"] =='PR'].sort_values(by="ESTIMATED_POP", ascending=False)
columns = ['CITY','STATE','CAPITAL','IDHM','LONG','LAT','ALT','AREA','GDP','GDP_CAPITA','ESTIMATED_POP']

df1 = df[columns]

corr = df1.corr()

plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

corrMat = plt.matshow(corr, fignum = 1)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.gca().xaxis.tick_bottom()

plt.colorbar(corrMat)

plt.title(f'Correlation Matrix', fontsize=15)

plt.show()
# remove zero values

mask1= df["IDHM"] != 0

mask2 = df["LAT"] !=0 

 

# create data

x = df[mask1&mask2]["IDHM"]

y = df[mask1&mask2]["LAT"]

z = df[mask1&mask2]["ESTIMATED_POP"]

 

# use the scatter function

plt.figure(figsize=(15, 8))

plt.title("HDI Human Development Index per Latitude per capita per Estimated Population 2018")

plt.xlabel("Latitude")

plt.ylabel("HDI Human Development Index")

plt.scatter(x, y, s=z/1000, alpha=0.5)

plt.show()
sns.scatterplot(x=df.IDHM, y=df.LAT)
plt.figure(figsize=(14, 8))

sns.scatterplot(x=df.LONG, y=df.LAT, size=df.IDHM)
plt.figure(figsize=(14, 8))

markers = {0:'o', 1:'s'}

sns.scatterplot(x=df.LONG, y=df.LAT, size=df.IDHM, hue=df.IDHM,style=df.CAPITAL, markers=markers)
f, ax = plt.subplots(figsize=(14, 8))

sns.scatterplot(x=df.LONG, y=df.LAT, size=df.IDHM, hue=df.IDHM)

sns.scatterplot(x=df[df.CAPITAL==1].LONG, y=df[df.CAPITAL==1].LAT, s=100)
# remove zero values

mask1= df["GDP_CAPITA"] != 0

mask2 = df["IDHM"] !=0 

 

# create data

x = df[mask1&mask2]["GDP_CAPITA"]

y = df[mask1&mask2]["IDHM"]

z = df[mask1&mask2]["ESTIMATED_POP"]

 

# use the scatter function

plt.figure(figsize=(15, 8))

plt.title("HDI Human Development Index per Gross Domestic Product per capita per Estimated Population 2018")

plt.xlabel("Gross Domestic Product")

plt.ylabel("HDI Human Development Index")

plt.scatter(x, y, s=z/5000, alpha=0.5)

plt.show()
plt.figure(figsize=(15, 10))

plt.hist(df["GDP_CAPITA"])

plt.show()
plt.figure(figsize=(12, 8))

sns.distplot(df.GDP_CAPITA)
mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 

mask3 = df["CATEGORIA_TUR"] != 0.



sns.lmplot( x="LONG", y="LAT", data=df[mask1&mask2&mask3], 

           fit_reg=False, hue='CATEGORIA_TUR', legend=True, scatter_kws={"s": 30},

          height=10)
mask1= df["GDP_CAPITA"] != 0

mask2 = df["BEDS"] !=0 



sns.lmplot( x="GDP_CAPITA", y="BEDS", data=df[mask1&mask2], 

           fit_reg=False, hue='UBER', legend=True, scatter_kws={"s": 30},

          height=7)