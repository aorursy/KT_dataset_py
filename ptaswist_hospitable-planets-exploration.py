import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.colors as clr

import matplotlib.cm as cm



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#read data 

planets = pd.read_csv('../input/oec.csv')

planets.head()
planets.shape
#display star mass, radius, metallicity and temperature as scatter plot

features = ['HostStarMassSlrMass','HostStarMetallicity','HostStarTempK', 'HostStarRadiusSlrRad']

stars_scatter = planets[features].dropna()

x = stars_scatter['HostStarMassSlrMass']

y = stars_scatter['HostStarMetallicity']

area = np.pi * stars_scatter['HostStarRadiusSlrRad']**2

colors = stars_scatter['HostStarTempK']



fig = plt.figure()

fig.suptitle('Known stars with exoplanets', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)

ax.set_xlabel('Solar mass')

ax.set_ylabel('Metallicity')



normalize = clr.Normalize(vmin=colors.min(), vmax=colors.max())

colormap = cm.coolwarm



#add colorbar showing temperature in K

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)

scalarmappaple.set_array(colors)

clb = plt.colorbar(scalarmappaple)

clb.set_label('Photosphere temperature [K]')





#mark the Sun on the plot

plt.scatter(x, y, s=area, c = normalize(colors), cmap = colormap, alpha=0.3)

plt.annotate('the Sun', xy = (1,0), 

             xytext = (-40, 40),

             textcoords = 'offset points', ha = 'right', va = 'bottom',

             bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue', alpha = 0.1),

             arrowprops = dict(arrowstyle = '->',connectionstyle = 'arc3,rad=0')

             )
stars_slr = planets[(planets.HostStarMassSlrMass > 0.5)&(planets.HostStarMassSlrMass < 1.5)&(planets.HostStarRadiusSlrRad > 0.5)&(planets.HostStarRadiusSlrRad < 1.5)]

star_features = ['HostStarMassSlrMass','HostStarRadiusSlrRad','HostStarMetallicity','HostStarTempK','HostStarAgeGyr']

stars_slr[star_features].describe()
print("Number of Sun-like stars: " + str(stars_slr.shape[0]))
#let's take a look at our new selection

stars_scatter_slr = stars_slr[features].dropna()

x = stars_scatter_slr['HostStarMassSlrMass']

y = stars_scatter_slr['HostStarMetallicity']

area = np.pi * stars_scatter_slr['HostStarRadiusSlrRad']**2

colors = stars_scatter_slr['HostStarTempK']



fig = plt.figure()

fig.suptitle('Known Sun-like stars with exoplanets', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)

ax.set_xlabel('Solar mass')

ax.set_ylabel('Metallicity')



normalize = clr.Normalize(vmin=colors.min(), vmax=colors.max())

colormap = cm.coolwarm



#add colorbar showing temperature in K

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)

scalarmappaple.set_array(colors)

clb = plt.colorbar(scalarmappaple)

clb.set_label('Photosphere temperature [K]')





#mark the Sun on the plot

plt.scatter(x, y, s=area, c = normalize(colors), cmap = colormap, alpha=0.3)

plt.annotate('the Sun', xy = (1,0), 

             xytext = (-50, 50),

             textcoords = 'offset points', ha = 'right', va = 'bottom',

             bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.3),

             arrowprops = dict(arrowstyle = '->',connectionstyle = 'arc3,rad=0')

             )
planet_features = ['PlanetIdentifier','PlanetaryMassJpt','RadiusJpt','PeriodDays','SemiMajorAxisAU','SurfaceTempK','AgeGyr']

#splitting up the filtering for readability

earth_plt = planets[(planets.PlanetaryMassJpt > 0.0015)&(planets.PlanetaryMassJpt < 0.03)&(planets.SurfaceTempK > 170)&(planets.SurfaceTempK < 350)]
earth_plt
planets_lum = pd.DataFrame.copy(planets)

#calculate star luminosity

planets_lum['Luminosity'] = planets_lum['HostStarRadiusSlrRad']**2  * (planets_lum['HostStarTempK']/5777)**4



#add habitable zone boundaries

planets_lum['HabZoneOut'] = np.sqrt(planets_lum['Luminosity']/0.53)

planets_lum['HabZoneIn'] = np.sqrt(planets_lum['Luminosity']/1.1)
planets_lum[['HostStarRadiusSlrRad','HostStarTempK','Luminosity','HabZoneOut','HabZoneIn']].head()
earth_plt2 = planets_lum[(planets_lum.PlanetaryMassJpt > 0.0015)&(planets_lum.PlanetaryMassJpt < 0.03)&(planets_lum.SemiMajorAxisAU > planets_lum.HabZoneIn)&(planets_lum.SemiMajorAxisAU < planets_lum.HabZoneOut)]

earth_plt2[planet_features]