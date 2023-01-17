import numpy as np
lots_of_data1 = np.arange(10)*2/57

lots_of_data2 = np.linspace(0,11,777)**2
#save your hard work (maybe something that was computationally intensive)

np.savez('save_my_progress.npz',mydata1=lots_of_data1,mydata2=lots_of_data2)
#you can come back later and re-load those right back in

reloaded = np.load('save_my_progress.npz')

reloaded['mydata1']
#Let's make a class to pickle

import pickle



class Vehicle:

    def __init__(

        self, 

        name = None,

        number_of_wheels = None,

        top_speed = None,

        ):

        

        self.name = name

        self.nwheels = number_of_wheels

        self.topspeed = top_speed

        self.currentspeed = 0

        

    def accelerate(self,how_much=1):

        self.currentspeed += how_much

        print('vroooooommmm')

        

mustang = Vehicle(

            name = 'Elanor',

            number_of_wheels = 4,

            top_speed = 200)

print(mustang.name)

mustang.accelerate()
#reminder that you may not be creating classes but you are definitely using them. For example:

arr = np.ones((100,2))

print(arr.dtype)

print(arr.sum(axis=1))
#We can pickle the class though, and get back what we put in

pickle.dump(mustang,open('vehicle.pkl','wb'))
myReadInOldVehichle = pickle.load(open('vehicle.pkl','rb'))

print(myReadInOldVehichle.name)
#Dictionary

pets_info = {

    'cat':{'color':'black','weight':10,'toys':['string','box','squeaky']},

    'dog':{'color':'white','weight':40,'toys':['ball','frisbee','squeaky']}

}
import json

print(json.dumps(pets_info, indent=4))
outfile = open('pets.json','w')

outfile.write(json.dumps(pets_info, indent=4))

outfile.close()
readbackin =open('pets.json','r').read()

dictionary_read_back_in = json.loads(readbackin)

dictionary_read_back_in['dog']
import numpy as np
a = np.zeros((2,2))   # Create an array of all zeros

print('zeros\n',a)              # Prints "[[ 0.  0.]

                      #          [ 0.  0.]]"



b = np.ones((1,2))    # Create an array of all ones

print('ones\n',b)              # Prints "[[ 1.  1.]]"



c = np.array([[1,1],[2,2]])  # Create a matrix

print('matrix\n',c)               

                       

d = np.eye(2)         # Create a 2x2 identity matrix

print('identity\n',d)              # Prints "[[ 1.  0.]

                      #          [ 0.  1.]]"



rando = np.random.random((5,3))  # Create an array filled with random values

print('random\n',rando)                     

                            

#you can set some/all elements at once

rando[3,:] = 0

print(rando)
#you can specify a list of rows/columns

rando[[1,2],:] = 0

print(rando)
print(rando.shape)
x = np.array([[1,2],[3,4]], dtype=np.float64)

y = np.array([[5,6],[7,8]], dtype=np.float64)



# Elementwise sum; both produce the array

# [[ 6.0  8.0]

#  [10.0 12.0]]

print(x + y)

print(np.add(x, y))
v = np.array([9,10])

w = np.array([11, 12])



# Inner product of vectors; both produce 219

print(v.dot(w))

print(np.dot(v, w))

x = np.array([[1,2],[3,4]])

v = np.array([9,10])



# Matrix / vector product; both produce the rank 1 array [29 67]

print(x.dot(v))

print(np.dot(x, v))
x = np.array([[1,2],[3,4]])

y = np.array([[5,6],[7,8]])



# Matrix / matrix product; both produce the rank 2 array

print(x.dot(y))

print(np.dot(x, y))
x = np.array([[1,2],[3,4]])



print(np.sum(x))  # Compute sum of all elements; prints "10"

print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"

print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
!ls ../input/
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)



fig = plt.figure()

plt.plot(x, np.sin(x), '-')

plt.plot(x, np.cos(x), '--');

#plt.savefig('test.pdf')
fig.canvas.get_supported_filetypes()
plt.savefig('test1.pdf')

!ls
rng = np.random.RandomState(0)

plt.figure(figsize=(12,8))

for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:

    plt.plot(rng.rand(5), rng.rand(5), marker, ms=13,

             label="marker=%s"%(marker))

plt.legend()

plt.xlim(0, 1.8);
rng = np.random.RandomState(0)

x = rng.randn(100)

y = rng.randn(100)

colors = rng.rand(100)

sizes = 1000 * rng.rand(100)



plt.figure(figsize=(12,8))

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,

            cmap='viridis')

plt.colorbar();  # show color scale

x = np.linspace(0, 10, 50)

dy = 0.8

y = np.sin(x) + dy * np.random.randn(50)



plt.figure(figsize=(12,8))

plt.errorbar(x, y, yerr=dy, fmt='o', color='black',

             ecolor='lightgray', elinewidth=3, capsize=3);

#Now lets read in some vehicle data

data = np.genfromtxt('../input/mpg-data/Auto.txt', names=True, usecols=range(7))

print(type(data))

print(data.dtype)
plt.figure(figsize=(14,11))  # create a plot figure



# create the first of two panels and set current axis

plt.subplot(2, 1, 1) # (rows, columns, panel number)

plt.scatter(data['horsepower'], data['acceleration'],color='blue',s=4)

plt.xlabel('Horespower')

plt.ylabel('Acceleration')



# create the second panel and set current axis

plt.subplot(2, 1, 2)

plt.scatter(data['horsepower'], data['mpg'],color='orange',s=4)

plt.xlabel('Horespower')

plt.ylabel('Miles Per Gallon');



# plt.subplot(3, 1, 3)

# plt.scatter(data['horsepower'], data['weight'])

# plt.xlabel('Horespower')

# plt.ylabel('Weight')

# #plt.tight_layout()
# First create a grid of plots

# ax will be an array of two Axes objects

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))

print(type(axes))

print(axes.shape)

# axes[0,0].scatter(data['horsepower'], data['acceleration'])

# axes[0,0].set_xlabel('Horsepower')

# plt.tight_layout()
#your code here
x = np.linspace(0, 10, num=11, endpoint=True)

y = np.cos(-x**2/9.0)

plt.plot(x,y)
from scipy.interpolate import interp1d

f = interp1d(x, y) # default is linear

f2 = interp1d(x, y, kind='cubic')

#notice that f and f2 are now functions!

f(5)
f([2,3,4])
xnew = np.linspace(0, 10, num=99) # lets make it smooth now

plt.figure(figsize=(10,6))

plt.plot(x, y, 'o', label='data')

plt.plot(xnew, f(xnew), '-', label='linear')

plt.plot(xnew, f2(xnew), '--', label='cubic')

plt.legend(loc='best', fontsize=16)

plt.show()
data = np.genfromtxt('../input/des-supernova-data-and-model/DARK_ENERGY_SURVEY_SUPERNOVA.txt', names=True)

xdat = data['zHD'] #velocity

ydat = data['MU'] #distance

ydaterr = data['MUERR']



matter_only_universe = np.genfromtxt('../input/des-supernova-data-and-model/decelerating_universe.txt', names=True, delimiter=',')

xmatter = matter_only_universe['zHD']

ymatter = matter_only_universe['MU']



dark_energy_filled_universe_model = np.genfromtxt('../input/des-supernova-data-and-model/accelerating_universe.txt', names=True, delimiter=',')

xaccel = dark_energy_filled_universe_model['zHD']

yaccel = dark_energy_filled_universe_model['MU']
plt.scatter(xdat,ydat,s=10)

plt.xlabel('Velocity')

plt.ylabel('Distance')
#your code here
x = np.random.normal(size=1000)



fig, ax = plt.subplots()



H = ax.hist(x, bins=50, alpha=0.5, histtype='stepfilled')
fig, ax = plt.subplots()



H = ax.hist(x, bins=np.arange(-3,4,1), alpha=0.5, histtype='step',linewidth=4, density=True) #density=normalized

from sklearn.datasets.samples_generator import make_blobs

n_components = 3

DATA, _ = make_blobs(n_samples=2000, centers=n_components, 

                      cluster_std = [5, 2.5, 2], 

                      random_state=42)

#Extract x and y

x = DATA[:,0]

y = DATA[:,1]
plt.scatter(x, y, s=10)

plt.title(f"Example of a mixture of {n_components} distributions")

plt.xlabel("x")

plt.ylabel("y");
# Create a figure with 6 plot areas

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))

 

# Everything sarts with a Scatterplot

a = axes[0].set_title('Scatterplot')

a = axes[0].scatter(x, y,s=10,color='black')

# As you can see there is a lot of overplottin here!

 

# Thus we can cut the plotting window in several hexbins

nbins = 20

a = axes[1].set_title('Hexbin')

a = axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

 

# 2D Histogram

a = axes[2].set_title('2D Histogram')

a = axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

def f1(x,y):

    from scipy.stats import norm

    Z = norm.pdf(X)*norm.pdf(Y)

    return Z
%matplotlib inline

x = np.linspace(-3, 3, 100)

y = np.linspace(-3, 3, 100)



X, Y = np.meshgrid(x, y)



Z = f1(X, Y)



plt.contour(X, Y, Z)

#plt.contourf(X, Y, Z,alpha=.5)
plt.contour(X, Y, Z, levels = [.01, .05, .1])
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

#%matplotlib notebook



# Create a surface plot and projected filled contour plot under it.

fig = plt.figure(figsize=(10,6))

ax = fig.gca(projection='3d')



ax.plot_surface(X, Y, Z, 

                cmap=cm.viridis)



cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(-0.15,0.2);

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

# 1. Draw the map background

fig = plt.figure(figsize=(8, 8))

#philadelphia

centerlat = 39.95

centerlon = -75.16

m = Basemap(projection='lcc', #resolution='h', 

            lat_0=centerlat, lon_0=centerlon,

            width=1E6, height=1.2E6)

m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawstates(color='gray')

m.scatter(centerlon,centerlat,s=1000,c='blue',latlon=True)

cities = np.genfromtxt('../input/california-cities/california_cities.csv',names=True,delimiter=',')



# Extract the data we're interested in

lat = cities['latd']

lon = cities['longd']

population = cities['population_total']

#your code here
import numpy as np

import matplotlib.pyplot as plt



#raw files are extremely efficient (binary data)

im = np.fromfile('../input/ctscan/ct.raw',dtype=np.uint16).reshape((512,512)) #need to specify the size and dtype to recover the original image





plt.figure(figsize=(8,8))



extent = (0, 10, 0, 10) #you get to define the axes

_ = plt.imshow(im, cmap=plt.cm.hot, extent=extent)



plt.scatter(6.8, 6, facecolors='none', edgecolors='blue', s=500)

plt.scatter(6.4, 5.9, facecolors='none', edgecolors='blue', s=500)

plt.text(5.8, 7.,'Something Bad?',color='white',rotation=30,fontsize=12);
# Lets do some edge detection for fun

from skimage import feature

edges1 = feature.canny(im.astype(float), sigma=8)

plt.imshow(edges1, cmap=plt.cm.gray)
import pandas as pd

import numpy as np
!head -10 ../input/mpg-data/Auto.txt
#Lets read in that old Auto data into a dataframe

df = pd.read_csv('../input/mpg-data/Auto.txt',delimiter='\t')
df.head(10)
df['horsepower'].plot.hist()
df.groupby('year').mean()

#This is a new table that we can plot!
df.groupby('year').mean()['horsepower'].plot(figsize=(10,7))

plt.ylabel('horsepower');
df.groupby('year').mean()['acceleration'].plot(figsize=(10,7))

plt.ylabel('acceleration');
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(16, 12), diagonal='hist');
color_palette = {0: "red", 1: "green"}

mycolors = [color_palette[int(c)] for c in df['year']>=75]   

pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(16, 12), diagonal='hist', color=mycolors);

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# We'll use a built in dataset from seaborn

# this requires that your kernel is connected to the internet

titanic = sns.load_dataset('titanic')

titanic.head()
# Let's start with a histogram of a single variable

sns.distplot(titanic['fare'],kde=False,color="red",bins=30)
# You can also joint plots: essentially histograms with two variables

sns.jointplot(x="fare",y="age",data=titanic,kind="scatter")
# Easily visualize categorical variables with boxplots, violin plots, or swarm plots

sns.boxplot(x="class",y="age", data=titanic)
# Make heatmaps - note that you need to transform your data into a matrix form first.

# The pandas fuction corr will make correlation matrices of numerical variables

sns.heatmap(titanic.corr(),cmap="coolwarm")
# One epecially useful plot for exploratory data analysis is a pair plot,

# which shows you distributions and scatterplots for all your variables



# we're going to select only a few variables so it is easier to see

titanic_subset = titanic[["survived","class","age","fare","sex","alone"]]

sns.pairplot(titanic_subset,hue='survived',palette='coolwarm')