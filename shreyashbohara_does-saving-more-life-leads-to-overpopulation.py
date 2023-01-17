#First lets look how data looks like

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



world_population = pd.read_csv("../input/world_population.csv")

world_population_growth = pd.read_csv("../input/world_population_annual_growth_rate.csv")
world_population.info()

world_population.head()
# Removing whitespace from population column

world_population.columns = world_population.columns.str.strip() 



# Removing whitespace in between numbers of population column

world_population['Population'] = world_population['Population'].str.replace(" ","")



# Making population column data from object to int64

world_population['Population'] = world_population['Population'].astype(str).astype(int)



world_population.info()

world_population.head()
world_population_growth.info()

world_population_growth.head()
# Since the y scale of our datasets has a big difference, that's why I divided population column by 10000000 so that it can fit in same scale

world_population['Population'] = world_population['Population']/10000000

world_population.head()
xticks = np.arange(1950, 2101, 10)

yticks = np.arange(0, 2.4, 0.2)



plt.figure(figsize=(16, 9), dpi=200)

plt.plot(world_population['Year'], world_population['Population'], label='Population', color="#c5d9fe")

plt.plot(world_population_growth['year'], world_population_growth['annual_growth_rate'],label='Annual Growth Rate of World Population', color="#c1003f")

#plt.rcParams['figure.figsize'] = (12,9)

plt.legend(loc='best')

plt.xlabel('Year')

plt.ylabel('Annual Growth of World Population Rate %')

plt.xticks(xticks)

plt.yticks(yticks)

#plt.autoscale(enable=True, axis='both', tight=None)

plt.grid(color='r', linestyle='-', alpha=0.1)

plt.text(1968, 2.1, '2.1%', fontweight='bold', bbox=dict(facecolor='#99fcc2', alpha=0.6))

plt.text(1950, 0.253628, '2.5 Billion', bbox=dict(facecolor='#99fcc2', alpha=0.6))

plt.text(1960, 0.370058, '3 Billion', bbox=dict(facecolor='#99fcc2', alpha=0.6))

plt.text(1980, 0.445841, '4.4 Billion', bbox=dict(facecolor='#99fcc2', alpha=0.6))

plt.text(2016, 0.746696, '7.4 Billion', bbox=dict(facecolor='#99fcc2', alpha=0.6))

plt.text(2040, 0.921034, '9.2 Billion', style='italic', bbox=dict(facecolor='#99fcc2', alpha=0.3))

plt.text(2060, 1.02226, '10.2 Billion', style='italic', bbox=dict(facecolor='#99fcc2', alpha=0.3))

plt.text(2080, 1.084871, '10.8 Billion',style='italic', bbox=dict(facecolor='#99fcc2', alpha=0.3))

plt.text(2100, 1.118437, '11.2 Billion',style='italic', bbox=dict(facecolor='#99fcc2', alpha=0.3))

plt.show()
