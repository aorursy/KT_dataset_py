from shapely.geometry import Point

t=Point(0.0,0.0)

print("Point Itself : ",t)

print("Area = ",t.area)

print("Length = ",t.length)

print("List of points : ",list(t.coords))
from shapely.geometry import LineString

line= LineString([(0,0),(2,3)])

list(line.coords)
from shapely.geometry import Polygon

pl=Polygon([(1,1), (2,1), (1,0)])

print(pl.area)
list(pl.exterior.coords)
%matplotlib inline

import matplotlib.pyplot as plt
x=[1,2,3,6,1]

y=[1,2,6,1,0]

fig, ax= plt.subplots()

ax.fill(x,y)
import numpy as np

#Fixing random state for reproducibility

np.random.seed(19680801)
N = 100

r0 = 0.9

x = 0.9 * np.random.rand(N)

y = 0.9 * np.random.rand(N)

area = (20 * np.random.rand(N))**2 # 0 to 10 point Radii

c=np.sqrt(area)

r=np.sqrt(x*x + y*y)

area1 = np.ma.masked_where(r<r0, area)

area2= np.ma.masked_where(r>=r0, area)

area1
plt.figure(figsize=(10,10))

plt.scatter(x,y,s=area1, marker='^',c=c)

plt.scatter(x,y,s=area2, marker='o',c=c)



# show the boundary between the regions

theta = np.arange(0, np.pi / 2, 0.01)

plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

plt.show()
plt.figure(figsize=(10,10))
plt.plot(x,x,label='linear')

plt.plot(x,x**2, label='quadratic')

plt.plot(x,x**3, label='cubic')



plt.xlabel('y label')

plt.ylabel('x label')
from matplotlib import pyplot

from shapely.geometry import LineString

from descartes import PolygonPatch
Blue = '#6699cc'

LIME_GREEN = '#32cd32'
#Shapely



line=LineString([(0,2),(5,6)])

b=line.buffer(0.3)
#matplotlib



fig=pyplot.figure(figsize=(15,6))

ax=fig.add_subplot(121)

x,y=line.xy

ax.plot(x,y,color=LIME_GREEN)
#We do not need descartes all the time, only someti,es when we need to plot patch on the line

fig= pyplot.figure(figsize=(15,6))

ax= fig.add_subplot(121)

ax.plot(x,y,color=LIME_GREEN)



#Descartes

patch1 = PolygonPatch(b,fc=Blue, ec= Blue, alpha= 0.2)

ax.add_patch(patch1)



#contextily

import contextily as ctx

#ctx.add_basemap(ax)

import cartopy.crs as ccrs
plt.figure(figsize=(8,8))

ax= plt.axes(projection= ccrs.AzimuthalEquidistant(central_latitude=90))

ax.coastlines(resolution='110m')

ax.gridlines()

import matplotlib.pyplot as plt



plt.figure(figsize=(12,12))

ax= plt.axes(projection= ccrs.PlateCarree())

ax.coastlines()



 #save the plot by calling plt.savefig() before plt.show()

     

plt.savefig('coastlines.pdf')

plt.savefig('coastlines.png')



plt.show()
import matplotlib.pyplot as plt

import cartopy.crs as ccrs



plt.figure(figsize=(12,9))

ax= plt.axes(projection= ccrs.RotatedPole(pole_latitude=37.5, pole_longitude= 177.5))

ax.coastlines(resolution='110m')

ax.gridlines()
import matplotlib.pyplot as plt

import cartopy.crs as ccrs



plt.figure(figsize=(10.59,10))

ax= plt.axes(projection= ccrs.Mercator())

ax.coastlines(resolution='110m')

ax.gridlines()
from shapely import shape, mapping

import fiona

# schema of the new shapefile

schema =  {'geometry': 'Polygon','properties': {'area': 'float:13.3','id_populat': 'int','id_crime': 'int'}}

# creation of the new shapefile with the intersection

with fiona.open('intersection.shp', 'w',driver='ESRI Shapefile', schema=schema) as output:

    for crim in fiona.open('crime_stat.shp'):

        for popu in fiona.open('population.shp'):

           if shape(crim['geometry']).intersects(shape(popu['geometry'])):     

              area = shape(crim['geometry']).intersection(shape(popu['geometry'])).area

              prop = {'area': area, 'id_populat' : popu['id'],'id_crime': crim['id']} 

              output.write({'geometry':mapping(shape(crim['geometry']).intersection(shape(popu['geometry']))),'properties': prop})

                
import shapely

help(shapely)