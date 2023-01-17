# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap



import plotly.graph_objs as go

import plotly.plotly as py

from plotly.graph_objs import *



import matplotlib.colors as colors

import matplotlib.cm as cm

import matplotlib.patches as mpatches



import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data.head()
#data['nkill'] = data['nkill'].replace(np.nan, 0)

#data['nwound'] = data['nwound'].replace(np.nan, 0)



data.dropna(subset=['nkill'])

data.dropna(subset=['nwound'])
plt.figure(figsize=(30,16))



var1 = data[(data.nkill>=0)&(data.nkill <=5)] 

var2 = data[(data.nkill>5)&data.nkill <=10]

var3 = data[data['nkill'] >10]



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()



# x, y = m(list(asia["longitude"].astype("float")), list(asia["latitude"].astype(float)))

# m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF", label = "Asia")



x, y = m(list(var1["longitude"].astype(float)), list(var1["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.9, color = 'green')



x, y = m(list(var2["longitude"].astype(float)), list(var2["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.2, color = 'yellow')



x, y = m(list(var3["longitude"].astype(float)), list(var3["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.2, color = 'red')





plt.title('Global Terror Attacks (1970-2015) - number of people killed', fontsize=35)

plt.legend(handles=[mpatches.Patch(color='green', label = "< 6 kills"),

                    mpatches.Patch(color='yellow',label='6 - 10 kills'), mpatches.Patch(color='red',label='> 10 kills')],fontsize=30, markerscale = 5)

    

plt.show()
plt.figure(figsize=(30,16))



var4 = data[(data.nwound>0)&(data.nwound <=10)] 

var5 = data[(data.nwound>10)&(data.nwound <=25)]

var6 = data[data['nwound'] >25]



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()



# x, y = m(list(asia["longitude"].astype("float")), list(asia["latitude"].astype(float)))

# m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF", label = "Asia")



x, y = m(list(var4["longitude"].astype(float)), list(var4["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.5, color = 'green')



x, y = m(list(var5["longitude"].astype(float)), list(var5["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.9, color = 'yellow')



x, y = m(list(var6["longitude"].astype(float)), list(var6["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.2, color = 'red')



plt.title('Global Terror Attacks (1970-2015) - number of people wounded', fontsize=35)

plt.legend(handles=[mpatches.Patch(color='green', label = "1 - 10 wounded"),

                    mpatches.Patch(color='yellow',label='10 - 25 wounded'), mpatches.Patch(color='red',label='> 25 wounded')],fontsize=30, markerscale = 5, loc=3)

plt.show()
plt.figure(figsize=(30,16))



var7 = data[(data.propextent==3)] 

var8 = data[(data.propextent==2)]

var9 = data[(data.propextent==1)]



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()



# x, y = m(list(asia["longitude"].astype("float")), list(asia["latitude"].astype(float)))

# m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF", label = "Asia")



x, y = m(list(var7["longitude"].astype(float)), list(var7["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = 'green', label = "1 - 3 kills")



x, y = m(list(var8["longitude"].astype(float)), list(var8["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = 'yellow', label = "4 - 5 kills")



x, y = m(list(var9["longitude"].astype(float)), list(var9["latitude"].astype(float)))

m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = 'red', label = "> 5 kills")





plt.title('Global Terror Attacks (1970-2015) - Property damage', fontsize=35)

plt.legend(handles=[mpatches.Patch(color='green', label = "< $1 million"),

                    mpatches.Patch(color='yellow',label='$ 1 million - $1 billion'), mpatches.Patch(color='red',label='> $ 1 billion')],fontsize=30, markerscale = 5, loc=3)

plt.show()