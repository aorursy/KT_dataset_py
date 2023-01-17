import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



import os

os.listdir('../input')
years=[1,1000,1500,1600,1700,1750,1800,1850,1900,1950,1955,1960,1965,1970,1980,1985,1990,

       1995,2000,2005,2010,2015]

pops=[200,400,458,580,682,791,1000,1262,1650,2525,2758,3018,3322,3682,

      4061,4440,4853,5310,5735,6127,6520,7349]

plt.plot(years,pops)

plt.show
years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]

pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]

plt.plot(years,pops,color=(255/255,100/255,100/255))

plt.ylabel("Population in Billions")

plt.xlabel("Population growth by year")

plt.title("Population Growth")

plt.show
x = [1,6,3]

y = [5,9,4]



x2 = [1,2,3]

y2 = [10,14,12]



plt.plot(x, y, label='First Line')

plt.plot(x2, y2, label='Second Line')

plt.xlabel('Plot Number')

plt.ylabel('Important var')

plt.title('Interesting Graph Check it out')

plt.legend()

plt.show()
years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]

pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]

death=[1.2,1.1,1.2,2.1,2.0,2.3,1.8,1.9,2.6,1.6,2.4,2.4,4.0]

plt.plot(years,pops,'--',color=(255/255,100/255,100/255))

plt.plot(years,death,color=(.6,.6,1))

plt.ylabel("Population in Billions")

plt.xlabel("Population growth by year")

plt.title("Population Growth")

plt.show
years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]

pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]

death=[1.2,1.1,1.2,2.1,2.0,2.3,1.8,1.9,2.6,1.6,2.4,2.4,4.0]

lines=plt.plot(years,pops,years,death)

plt.grid(True)

plt.setp(lines,color=(1,.4,.5),marker='*')

plt.show
labels=['Python','C','C++','PHP','Java','Ruby']

sizes=[33,52,12,17,42,48]

separated=(.1,0,0,0,0,0)

plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=separated)

plt.axis('equal')

plt.show()
raw_data={'names':['Nick','Sani','John','Rubi','Maya'],

         'jan_ir':[123,124,125,126,128],

         'feb_ir':[23,24,25,27,29],

         'march_ir':[3,5,7,6,9]}



df=pd.DataFrame(raw_data,columns=['names','jan_ir','feb_ir','march_ir'])

df
raw_data={'names':['Nick','Sani','John','Rubi','Maya'],

         'jan_ir':[123,124,125,126,128],

         'feb_ir':[23,24,25,27,29],

         'march_ir':[3,5,7,6,9]}



df=pd.DataFrame(raw_data,columns=['names','jan_ir','feb_ir','march_ir'])

df['total_ir']=df['jan_ir']+df['feb_ir']+df['march_ir']

df
color=[(1,.4,.4),(1,.6,1),(.5,.3,1),(.3,1,.5),(.7,.7,.2)]

plt.pie(df['total_ir'],labels=df['names'],colors=color,autopct='%1.1f%%')

plt.axis('equal')

plt.show()
korea_scores=(554,536,538)

canada_scores=(518,523,525)

china_scores=(413,570,580)

franch_scores=(495,505,499)

index=np.arange(3)

bar_width=.2

k1=plt.bar(index,korea_scores,bar_width,alpha=.9,label="Korea")

c1=plt.bar(index+bar_width,canada_scores,bar_width,alpha=.9,label="Canada")

ch1=plt.bar(index+bar_width*2,china_scores,bar_width,alpha=.9,label="China")

f1=plt.bar(index+bar_width*3,franch_scores,bar_width,alpha=.9,label="Franch")

plt.xticks(index+.6/2,('Mathematics','Reading','Science'))

plt.ylabel('Mean score in PISA in 2012')

plt.xlabel('Subjects')

plt.title('Test scores by Country')

plt.grid(True)

plt.legend()

plt.show()
plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")

plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')

plt.legend()

plt.xlabel('bar number')

plt.ylabel('bar height')



plt.title('Epic Graph\nAnother Line! Whoa')



plt.show()
population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]



bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]



plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)



plt.xlabel('x')

plt.ylabel('y')

plt.title('Interesting Graph\nCheck it out')

plt.show()
x = [1,2,3,4,5,6,7,8]

y = [5,2,4,2,1,4,5,2]



plt.scatter(x,y, label='skitscat Raggedy', color='k', s=25, marker="o")



plt.xlabel('x')

plt.ylabel('y')

plt.title('Interesting Graph\nCheck it out')

plt.legend()

plt.show()
days = [1,2,3,4,5]



sleeping = [7,8,6,11,7]

eating =   [2,3,4,3,2]

working =  [7,8,7,2,2]

playing =  [8,5,7,8,13]

plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])



plt.xlabel('x')

plt.ylabel('y')

plt.title('Interesting Graph\nStack Plots')

plt.show()
days = [1,2,3,4,5]



sleeping = [7,8,6,11,7]

eating =   [2,3,4,3,2]

working =  [7,8,7,2,2]

playing =  [8,5,7,8,13]





plt.plot([],[],color='m', label='Sleeping', linewidth=5)

plt.plot([],[],color='c', label='Eating', linewidth=5)

plt.plot([],[],color='r', label='Working', linewidth=5)

plt.plot([],[],color='k', label='Playing', linewidth=5)



plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])



plt.xlabel('x')

plt.ylabel('y')

plt.title('Interesting Graph\nCheck it out')

plt.legend()

plt.show()
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

from matplotlib import style



style.use('fivethirtyeight')



fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')



x = [1,2,3,4,5,6,7,8,9,10]

y = [5,6,7,8,2,5,6,3,7,2]

z = [1,2,6,3,2,7,3,3,7,2]



ax1.plot(x,y,z)



ax1.set_xlabel('x axis')

ax1.set_ylabel('y axis')

ax1.set_zlabel('z axis')



plt.show()
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

from matplotlib import style



style.use('ggplot')



fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')



x = [1,2,3,4,5,6,7,8,9,10]

y = [5,6,7,8,2,5,6,3,7,2]

z = [1,2,6,3,2,7,3,3,7,2]



x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]

y2 = [-5,-6,-7,-8,-2,-5,-6,-3,-7,-2]

z2 = [1,2,6,3,2,7,3,3,7,2]



ax1.scatter(x, y, z, c='g', marker='o')

ax1.scatter(x2, y2, z2, c ='r', marker='o')



ax1.set_xlabel('x axis')

ax1.set_ylabel('y axis')

ax1.set_zlabel('z axis')



plt.show()
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

import numpy as np

from matplotlib import style

style.use('ggplot')



fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')



x3 = [1,2,3,4,5,6,7,8,9,10]

y3 = [5,6,7,8,2,5,6,3,7,2]

z3 = np.zeros(10)



dx = np.ones(10)

dy = np.ones(10)

dz = [1,2,3,4,5,6,7,8,9,10]



ax1.bar3d(x3, y3, z3, dx, dy, dz)





ax1.set_xlabel('x axis')

ax1.set_ylabel('y axis')

ax1.set_zlabel('z axis')



plt.show()
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

import numpy as np

from matplotlib import style

style.use('ggplot')



fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')



x, y, z = axes3d.get_test_data()



ax1.plot_wireframe(x,y,z, rstride = 3, cstride = 3)



ax1.set_xlabel('x axis')

ax1.set_ylabel('y axis')

ax1.set_zlabel('z axis')



plt.show()
img = mpimg.imread('../input/unlabeled_images/unlabeled_image_png_30319.png')

print(img)
plt.figure(figsize=(10,6))

imgplot = plt.imshow(img)
plt.figure(figsize=(10,6))

lum_img = img[:, :, 0]

plt.imshow(lum_img)
plt.figure(figsize=(10,6))

plt.imshow(lum_img, cmap="hot")
plt.figure(figsize=(10,6))

imgplot = plt.imshow(lum_img)

imgplot.set_cmap('nipy_spectral')
plt.figure(figsize=(10,6))

imgplot = plt.imshow(lum_img)

plt.colorbar()
plt.figure(figsize=(10,6))

plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.figure(figsize=(10,6))

imgplot = plt.imshow(lum_img, clim=(0.0, 0.7))
fig = plt.figure(figsize=(10,6))

a = fig.add_subplot(1, 2, 1)

imgplot = plt.imshow(lum_img)

a.set_title('Before')

plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

a = fig.add_subplot(1, 2, 2)

imgplot = plt.imshow(lum_img)

imgplot.set_clim(0.0, 0.7)

a.set_title('After')

plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
from PIL import Image



img = Image.open('../input/unlabeled_images/unlabeled_image_png_51042.png')

img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place

imgplot = plt.imshow(img)
imgplot = plt.imshow(img, interpolation="nearest")

imgplot = plt.imshow(img, interpolation="bicubic")