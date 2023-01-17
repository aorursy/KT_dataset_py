# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as pl

import scipy.stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline

import plotly.graph_objects as go



import seaborn as sns

cardata = pd.read_csv("../input/craigslist-carstrucks-data/vehicles.csv", nrows=2000)



cardata.drop(["url", "region_url", "model", "condition", "cylinders", "drive", "size", "type", "paint_color", "image_url", "description", "lat", "long","county","vin"], axis = 1, inplace = True)



cardata.head()
cardata = cardata[cardata.price != 0]

cardata = cardata[cardata.odometer != 0]

cardata = cardata[cardata.odometer < 400000]



# This is cleaning my data to account for listings where the seller input an error or did not list a necessary data point

# This may cause the conclusions made to be less representitive of all listings, creating a bias in results
cardata.head()
cardata.describe()
plt.scatter(cardata['year'], cardata['price'])
plt.title('Price of Craigslist Car Sales',fontsize=10)



plt.xlabel('Price in Dollars')



plt.hist(cardata['price'], bins=20)
x_min = 0.0

x_max = 60000



mean = np.mean(cardata, axis=0) 

std = np.std(cardata, axis=0)



x = np.linspace(x_min, x_max, 100)



y = scipy.stats.norm.pdf(x,mean[1],std[1])



plt.plot(x,y, color='coral')



plt.grid()



plt.xlim(x_min,x_max)

plt.ylim(0,0.00005)



plt.title('Normal Distribution of Price of Craigslist Car Sales',fontsize=10)



plt.xlabel('Price in Dollars')

plt.ylabel('Normal Distribution')



plt.show()
plt.title('Year of Craigslist Car Sales',fontsize=10)



plt.xlabel('Year')



plt.hist(cardata['year'], bins = 30)
x_min = 1970

x_max = 2021



mean = np.mean(cardata, axis=0) 

std = np.std(cardata, axis=0)



x = np.linspace(x_min, x_max, 100)



y = scipy.stats.norm.pdf(x,mean[2],std[2])



plt.plot(x,y, color='coral')



plt.grid()



plt.xlim(x_min,x_max)

plt.ylim(0,0.07)



plt.title('Normal Distribution of Year of Craigslist Car Sales',fontsize=10)



plt.xlabel('Year')

plt.ylabel('Normal Distribution')



plt.show()
x_min = 1970

x_max = 2021



meanM = np.mean(cardata[cardata.transmission == "manual"], axis=0) 

stdM = np.std(cardata[cardata.transmission == "manual"], axis=0)

                      

meanA = np.mean(cardata[cardata.transmission == "automatic"], axis=0) 

stdA = np.std(cardata[cardata.transmission == "automatic"], axis=0)



x = np.linspace(x_min, x_max, 100)



yM = scipy.stats.norm.pdf(x,meanM[2],stdM[2])

yA = scipy.stats.norm.pdf(x,meanA[2],stdA[2])



plt.plot(x,yM, color='coral')

plt.plot(x,yA, color='green')



plt.grid()



plt.xlim(x_min,x_max)

plt.ylim(0,0.07)



plt.title('Normal Distribution of Year of Craigslist Car Sales: Manual Vs. Automatic',fontsize=10)



plt.xlabel('Year')

plt.ylabel('Normal Distribution')



plt.show()
plt.title('Odometer of Craigslist Car Sales',fontsize=10)



plt.xlabel('Miles')

plt.hist(cardata['odometer'], bins=20)
x_min = 0.0

x_max = 300000



mean = np.mean(cardata, axis=0) 

std = np.std(cardata, axis=0)



x = np.linspace(x_min, x_max, 100)



y = scipy.stats.norm.pdf(x,mean[3],std[3])



plt.plot(x,y, color='coral')



plt.grid()



plt.xlim(x_min,x_max)

plt.ylim(0,0.00001)



plt.title('Normal Distribution of Odometer of Craigslist Car Sales',fontsize=10)



plt.xlabel('Miles')

plt.ylabel('Normal Distribution')



plt.show()
print (mean)
