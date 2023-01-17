# import pandas and numpy

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from string import ascii_letters





# import the os package for Kaggle

import os

print(os.listdir("../input"))
# create dataframe

df = pd.read_csv("../input/BRAZIL_CITIES.csv", sep=";", decimal=",")
# view dataframe

df.sample(25)
# get dataframe shape

df.shape
# create backup copy

df_copy = df.copy(True)
''' select the columns you're interested in exploring. These are the one's I chose. 

if a model was being generated for some purpose, feature selection methods should be employed,

however, I will just being doing cleaning, vis and exploration '''



columns = ['CITY', 'STATE', 'CAPITAL', 'IBGE_RES_POP', 'IBGE_RES_POP_BRAS','IBGE_RES_POP_ESTR','IBGE_DU','IBGE_DU_URBAN','IBGE_DU_RURAL', 'IBGE_POP','IBGE_1','IBGE_1-4','IBGE_5-9','IBGE_10-14','IBGE_15-59','IBGE_60+','IBGE_PLANTED_AREA','IDHM','LONG','LAT','ALT','ESTIMATED_POP','GDP_CAPITA','Cars','Motorcycles','UBER','MAC','WAL-MART','BEDS']
# create reduced dataframe and check shape

r_df = df[columns]

r_df.shape
# create a seaborn pairplot

pp = sns.pairplot(r_df)
# the pairplot is still huge and difficult to find individual relationships in, let's try a correlation matrix for narrowing down our search

corr = r_df.corr()
# I prefer one sided matricies so create a mask

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# set up figure 

f, ax = plt.subplots(figsize=(15, 15))



cmap = sns.diverging_palette(220, 20, as_cmap=True)



sns.heatmap(corr, mask=mask,cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# let's look at what's in the uber column

r_df.UBER
# there are a lot of nans, possibly in place of zeros, let's check

df_copy.UBER.value_counts()
# I was right! let's fix that

r_df.UBER.replace({np.nan:0}, inplace=True)

r_df.UBER.value_counts()
# let's see if that does anything to our matrix

corr = r_df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(15, 15))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask,cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# let's investigate the strongest correlation first

sns.set_style('dark')

sns.scatterplot(x=r_df.IDHM, y=r_df.LAT)
# map of lat and long with IDHM detemining size

f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=r_df.LONG, y=r_df.LAT, size=r_df.IDHM)
# it's hard to see any trends here, let's add color to get a better idea

f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=r_df.LONG, y=r_df.LAT, size=r_df.IDHM, hue=r_df.IDHM)
# let's see if we can spot any capitals in there

f, ax = plt.subplots(figsize=(8, 8))

markers = {0:'o', 1:'s'}

sns.scatterplot(x=r_df.LONG, y=r_df.LAT, size=r_df.IDHM, hue=r_df.IDHM,style=r_df.CAPITAL, markers=markers)
# that's not helpful, let's try an overlay    

f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=r_df.LONG, y=r_df.LAT, size=r_df.IDHM, hue=r_df.IDHM)

sns.scatterplot(x=r_df[r_df.CAPITAL==1].LONG, y=r_df[r_df.CAPITAL==1].LAT, s=100)
# very cool! let's give GDP per capita the same treatment real quick

f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=r_df.LONG, y=r_df.LAT, size=r_df.GDP_CAPITA, hue=r_df.GDP_CAPITA)
# that's not what I expected, let's look at the data directly

r_df.GDP_CAPITA

# looks like the color palette binned the values,possibly obscuring some insights
# let's take a look at the distribution, after taking care of nans

f, ax = plt.subplots(figsize=(12, 8))

gdp = r_df.GDP_CAPITA.dropna()

sns.distplot(gdp)
# it looks like gdp is heavily right skewed with a massive tail. 

# it seems likely that those massive outliers are errors, and could be removed in some cases

gdp.describe()
# let's look at how UBER is doing

f, ax = plt.subplots(figsize=(12, 8))

sns.countplot(r_df['UBER'])

f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=r_df[r_df.UBER==0].LONG, y=r_df[r_df.UBER==0].LAT)

sns.scatterplot(x=r_df[r_df.UBER==1].LONG, y=r_df[r_df.UBER==1].LAT)
# let's see how cars in cities with and without uber look

f, ax = plt.subplots(figsize=(16, 12))

sns.boxplot(y=r_df['Cars'], x=r_df['UBER'])
# can removing some of the really high outlier give us a better picture

f, ax = plt.subplots(figsize=(16, 12))

ubers, car_vals = r_df[r_df.Cars <100000].UBER, r_df[r_df.Cars <100000].Cars

sns.boxplot(ubers, car_vals )

# there are way more cars in cities with uber
# comparing distributions of cars with and without UBER

f, ax = plt.subplots(figsize=(16, 12))

sns.distplot(r_df[(r_df.Cars < 100000) & (r_df.UBER==0)].Cars)

sns.distplot(r_df[(r_df.Cars < 100000) & (r_df.UBER==1)].Cars, bins=20)