%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import math
plt.style.use('ggplot')
data = pd.read_csv('../input/meteorite-landings.csv')

data.head()
data['year'] = data.dropna()['year'].astype(np.int64)
years = data.year.unique()

landings = []



for year in years:

    landings.append(len(data[data.year == year]))

    

s = [x for x in landings]



plt.scatter(years, landings, s=s, c='r', edgecolors='black')

plt.xlabel('Year')

plt.ylabel('Landings')

plt.ylim(ymin=-100)
m = Basemap()

x, y = m(data.reclong, data.reclat)

m.etopo()

m.scatter(x, y)
data[data.fall == "Fell"]