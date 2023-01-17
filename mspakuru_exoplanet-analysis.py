# Load in the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import pylab as pl
# Bring in the data

big_df = pd.read_csv("../input/oec.csv")



# Take a subset to look at method and year of discovery.

df = big_df[["DiscoveryMethod", "DiscoveryYear"]]

# Zap rows without useful data

df = df.dropna()
# Separate by Year then Method to obtain counts, then separate methods in order to plot them

byYearThenMethod = df['DiscoveryMethod'].groupby([df['DiscoveryYear'], df['DiscoveryMethod']]).count()

byYearRV = []

byYearImaging = []

byYearMicrolensing = []

byYearTiming = []

byYearTransit = []

# This double groupby creates a "multi-index", which one accesses like an array:

for row in byYearThenMethod.index:

    if row[1] == 'RV':

        byYearRV.append([row[0], byYearThenMethod[row]])

    if row[1] == 'imaging':

        byYearImaging.append([row[0], byYearThenMethod[row]])

    if row[1] == 'microlensing':

        byYearMicrolensing.append([row[0], byYearThenMethod[row]])

    if row[1] == 'timing':

        byYearTiming.append([row[0], byYearThenMethod[row]])

    if row[1] == 'transit':

        byYearTransit.append([row[0], byYearThenMethod[row]])        
# Plot the methods together, in order to compare them

plt.grid()

plt.plot(*zip(*byYearRV), label='RV')

plt.plot(*zip(*byYearImaging), label='Imaging')

plt.plot(*zip(*byYearMicrolensing), label='Microlensing')

plt.plot(*zip(*byYearTiming), label='Timing')

plt.plot(*zip(*byYearTransit), label='Transit')

plt.legend(loc='upper left')
# Make barchart of discoveries by year

byYear= df['DiscoveryYear'].groupby([df['DiscoveryYear']]).count()

y = byYear

x = byYear.index

width = 1/1.5

plt.grid()

plt.xlabel("Year")

plt.ylabel("Count")

plt.title("Exoplanets Discovered by Year")

xlabels = byYear.index

plt.bar(x, y, width, color="blue")
big_df.columns



# Take a subset to look at method and year of discovery.

df2 = big_df[["PlanetaryMassJpt", "RadiusJpt", "DistFromSunParsec", "DiscoveryYear"]]

# Zap rows without useful data

df2 = df2.dropna()
# Make scatterplot of mean mass of planets in comparison to Jupiter by year

mass = df2['PlanetaryMassJpt'].groupby([df2['DiscoveryYear']]).mean()

radius = df2['RadiusJpt'].groupby([df2['DiscoveryYear']]).mean()

massY = mass

massX = mass.index

width = 1/1.5

plt.grid()

plt.xlabel("Year")

plt.ylabel("Mean")

plt.title("Mean Mass of Exoplanets Discovered by Year")

xlabels = mass.index

plt.scatter(massX, massY, width, color="blue")

plt.plot(np.unique(massX), np.poly1d(np.polyfit(massX, massY, 1))(np.unique(massX)))
# Make scatterplot of mean radii of exoplanets in comparison to Jupiter by year

radius = df2['RadiusJpt'].groupby([df2['DiscoveryYear']]).mean()

radiusY = radius

radiusX =radius.index

width = 1/1.5

plt.grid()

plt.xlabel("Year")

plt.ylabel("Mean")

plt.title("Mean Radius of Exoplanets Discovered by Year")

xlabels = radius.index

plt.scatter(radiusX, radiusY, width, color="blue")

plt.plot(np.unique(radiusX), np.poly1d(np.polyfit(radiusX, radiusY, 1))(np.unique(radiusX)))
1/11
# Make scatterplot of mean distances of exoplanets by year

dist = df2['DistFromSunParsec'].groupby([df2['DiscoveryYear']]).mean()

distY = dist

distX =dist.index

width = 1/1.5

plt.grid()

plt.xlabel("Year")

plt.ylabel("Mean")

plt.title("Mean Distance of Exoplanets Discovered by Year")

xlabels = dist.index

plt.scatter(distX, distY, width, color="blue")

plt.plot(np.unique(distX), np.poly1d(np.polyfit(distX, distY, 1))(np.unique(distX)))