from mpl_toolkits.basemap import Basemap

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os





inDir = os.path.join("..", 'input')

outDir= os.path.join("..", 'output')

CompanyName="INFOSYS LIMITED"

#Read Data

df = pd.read_csv(os.path.join(inDir, 'h1b_kaggle.csv'))

#Select Company

Company=df[df.EMPLOYER_NAME.str.contains(CompanyName)==True]

#Get lat and lons from the company

lons=list(Company.lon.fillna(0))

lats=list(Company.lat.fillna(0))



# create the figure and axes instances

fig = plt.figure()

ax = fig.add_axes([0.1,0.1,0.8,0.8])

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,\

            resolution='c',area_thresh=1000.,projection='lcc',\

            lat_1=33,lat_2=45,lon_0=-95)



# draw coastlines and political boundaries

#------------other useful plotting commands------------

#m.fillcontinents(color="#DCE8E8", lake_color="#A7DCDA")

#m.drawrivers(color="blue")

#m.nightshade(datetime.datetime.now())

#------------------------------------------------------

m.etopo()

m.drawcoastlines()

m.drawcountries(linewidth=2)

m.drawstates()



# draw parallels and meridians

parallels = np.arange(0.,80,20.)

m.drawparallels(parallels,labels=[1,0,0,1])

meridians = np.arange(10.,360.,30.)

m.drawmeridians(meridians,labels=[1,0,0,1])



# add points

x,y = m(lons, lats)

m.plot(x, y, 'o', markersize=2, color="red")



ax.set_title('INFOSYS LIMITED Applications (2011-2016)')

plt.show()