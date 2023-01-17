# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv")

df=data

data.head()
indiaLang=pd.DataFrame(data[data["Countries"]=="India"])

indiaLang.head()
data["Degree of endangerment"].value_counts()
data["Degree of endangerment"][data["Degree of endangerment"]=="Vulnerable"]=0

data["Degree of endangerment"][data["Degree of endangerment"]=="Definitely endangered"]=0.25

data["Degree of endangerment"][data["Degree of endangerment"]=="Severely endangered"]=0.5

data["Degree of endangerment"][data["Degree of endangerment"]=="Critically endangered"]=0.75

data["Degree of endangerment"][data["Degree of endangerment"]=="Extinct"]=1

plt.scatter(

data["Longitude"],data["Latitude"],c=data["Degree of endangerment"],s=data["Number of speakers"]/10000,alpha=0.5)

plt.xlabel("Latitude")

plt.ylabel("Longitude")

plt.show()
lats=np.array(data["Latitude"].astype(float))

lons=np.array(data["Longitude"].astype(float))

plt.figure(figsize=(30,15))

map = Basemap()

map.fillcontinents(color='white',lake_color='lightblue',zorder=0.5)

map.drawmapboundary(fill_color='#C5EFF7')

#map.drawcountries()

map.drawcoastlines()

x, y = map(lons, lats)

map.scatter(x, y,c=data["Degree of endangerment"],s=data["Number of speakers"]/2000,alpha=0.6)

#plt.legend(df["Degree of endagerment"], )

plt.show()
