import requests

from pandas.io.json import json_normalize

import pandas as pd

import missingno as msno 

import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt

from scipy import stats

import time

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)

pd.set_option('display.width', 1000)
dfall = pd.read_csv("../input/cornell-car-rental-dataset/CarRentalDataV1.csv")
cols = ['fuelType', 'rating', 'renterTripsTaken', 'reviewCount', 'location.city', 'location.country', 

        'location.latitude', 'location.longitude', 'location.state',  'owner.id', 

        'rate.daily', 'vehicle.make','vehicle.model', 'vehicle.type', 'vehicle.year', 'airportcity']
sns.set(font_scale=1.5)

sns.set_context("notebook")

#sns.set_color_codes("dark")

sns.set(style="ticks")

a4_dims = (14.00, 7.80)

x = dfall['rate.daily']

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.distplot(x, bins=400, 

             kde=True, 

             color='gray', 

             hist_kws={"alpha":None, "color":'steelblue'}

            )

plt.xlim(0, 405)

ax.set_ylabel('no. of vehicles in each price range', fontsize=14)

ax.set_xlabel('daily rate (dollars per day)', fontsize=14)

sns.despine(trim=True, left=True)

ax.xaxis.grid(True)

plt.savefig('3rateVsCount.jpg', format='jpg')

dfall['count']=dfall['vehicle.make'].value_counts()
sns.set_context("notebook")

sns.set_color_codes("dark")

sns.set(style="white")

a4_dims = (20.00, 7.80)

fig, ax = pyplot.subplots(figsize=a4_dims)

plt.xticks(rotation=65)



sns.countplot(x='vehicle.make', data=dfall,

              order = dfall['vehicle.make'].value_counts().index,

              #hue='vehicle.year'

              palette="Blues_r" #"BrBG" #"cubehelix" #"GnBu_d"    #"BuGn_r"     #"Set2"

           )

ax.set_ylabel('count', fontsize=14, color='b')

ax.set_xlabel('make of the vehicle', fontsize=14, color='b')

plt.savefig('carVsCount.png', format='png')

sns.despine(left=True, bottom=True)
df2 = pd.melt(dfall, "vehicle.make", var_name="rate.daily")



sns.set_context("notebook")

#sns.set_color_codes("dark")

sns.set(style="whitegrid")

a4_dimsRel = (20.00, 9.80)

fig, ax = pyplot.subplots(figsize=a4_dimsRel)



sns.swarmplot(y="rate.daily", x="vehicle.make", 

              color="steelblue",

              data=dfall,

              order = dfall['vehicle.make'].value_counts().iloc[:20].index,

              size=3, 

              #color=".1", 

              linewidth=0)



ax.xaxis.grid(True)

ax.yaxis.grid(False)

#ax.set(ylabel="")

sns.despine(left=True, bottom=True)



plt.ylim(0, 405)

plt.xticks(rotation=30, fontsize=16)

#sns.despine(left=True, bottom=True)

ax.set_xlabel('Make of the vehicle', fontsize=18)

ax.set_ylabel('daily rate (dollars per day)', fontsize=18)

plt.savefig('carVsRate.jpg', format='jpg')

sns.set(style="ticks")

sns.set_context("notebook")

a4_dimsRel2 = (9.80, 12.80)

fig, ax = pyplot.subplots(figsize=a4_dimsRel2)

sns.stripplot(x="rate.daily", y="vehicle.make",

                #hue="kind",

                data=dfall,

                palette="husl", 

                size=5, 

                marker="D",

                edgecolor="gray", 

                alpha=.75,

                #linewidth=.5,

                #jitter=0.1,

                order = dfall['vehicle.make'].value_counts().index

              

                )

ax.xaxis.grid(True)

ax.yaxis.grid(False)

#ax.set(ylabel="")

sns.despine(trim=False, 

            left=True, bottom=False)



plt.xlim(0, 405)

ax.set_ylabel('Make of the vehicle', fontsize=14)

ax.set_xlabel('daily rate (dollars per day)', fontsize=14)





plt.savefig('5modelVsRate.png', format='png', bbox_inches="tight")
sns.set(style="ticks")

sns.set_context("notebook")

a4_dimsRel2 = (9.80, 12.80)

fig, ax = pyplot.subplots(figsize=a4_dimsRel2)



cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)



sns.stripplot(x="rate.daily", y="vehicle.model", data=dfall,

                order = dfall['vehicle.model'].value_counts().iloc[:50].index  #.iloc[:30] before index

                )



ax.xaxis.grid(True)

ax.yaxis.grid(False)

#ax.set(ylabel="")

sns.despine(trim=False, 

            left=True, bottom=False)





#plt.ylim(0,150)

plt.xlim(0, 400)

plt.yticks(fontsize=10)

ax.set_ylabel('Model of the vehicle', fontsize=14)

ax.set_xlabel('daily rate (dollars per day)', fontsize=14)



plt.savefig('6modelVsRate.png', format='png', bbox_inches="tight")
sns.set_context("notebook")

a4_dimsRel = (20.00, 9.80)

fig, ax = pyplot.subplots(figsize=a4_dimsRel)

sns.set(style="white")

# Draw a categorical scatterplot to show each observation

sns.boxplot(y="rate.daily", x="airportcity",  

              data=dfall.groupby('airportcity').filter(lambda x: len(x) >= 20),

              #order = dfall['airportcity'].value_counts().iloc[:20].index

              order = dfall.groupby('airportcity').filter(lambda x: len(x) >= 20)\

                     .groupby(['airportcity']).median()\

                    .sort_values('rate.daily', ascending = False)\

                    #.iloc[0:50]

                     .index)



ax.xaxis.grid(True)

ax.yaxis.grid(False)

sns.despine(left=True, bottom=True)



plt.ylim(0, 255)

plt.xticks(rotation=75, fontsize=14)

ax.set_xlabel('City', fontsize=14)

ax.set_ylabel('daily rate (dollars per day)', fontsize=14)



plt.savefig('4cityVsRate.png', format='png', bbox_inches="tight")

sns.set(style="darkgrid")

#sns.set_context("notebook")

a4_dims = (15, 12)

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.scatterplot(data=dfall, x='renterTripsTaken', y='rate.daily', 

                palette="ch:2,r=.1,l=.5_r",

                   size=1)

plt.xlim(0, 205)

plt.ylim(0,405)

ax.set_ylabel('rate', fontsize=14, color='b')

ax.set_xlabel('renterTripsTaken', fontsize=14, color='b')
dfall['revenue'] = dfall.apply(lambda x: x.loc['rate.daily']*x.loc['renterTripsTaken'], axis=1)
sns.set(style="darkgrid")

a4_dims = (15, 12)

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.scatterplot(data=dfall, x='revenue', y='rate.daily', 

                palette="ch:2,r=.1,l=.5_r")

plt.ylim(0,405)

ax.set_ylabel('rate', fontsize=14, color='b')

ax.set_xlabel('revenue', fontsize=14, color='b')

ax.set_xscale('log')

a4_dims = (15, 8)

x = dfall['rate.daily']

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.kdeplot(x, dfall.renterTripsTaken, shade=False, shade_lowest=False, cmap='Purples_d',  n_levels=20)

plt.xlim(0, 250)

plt.ylim(-20,100)

ax.set_ylabel('no. of Trips Taken', fontsize=14, color='b')

ax.set_xlabel('daily rate (dollars per day)', fontsize=14, color='b')