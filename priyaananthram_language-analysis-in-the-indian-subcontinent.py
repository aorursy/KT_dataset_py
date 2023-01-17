# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import numpy as np

%matplotlib inline



df = pd.read_csv('../input/data.csv', usecols=[0, 1,4, 5, 7, 10, 12, 13])

df = df.rename(

    columns={'Name in English':'language', 'Country codes alpha 3':'locations','Countries':'countries',

             'Degree of endangerment':'risk', 'Number of speakers':'population'})

df.columns = df.columns.str.lower()



subcontinent=df.loc[df.locations.isin(['IND','BTN','NPL','BGD','PAK','LKA','MMR'])]

india = df[df.countries == "India"]



bhutan= df[df.locations == "BTN"]

nepal= df[df.locations== "NPL"]

bangla= df[df.locations == "BGD"]

pak= df[df.locations == "PAK"]

burma= df[df.locations == "MMR"]



sri= df[df.locations == "LKA"]

india.head()
india.risk.value_counts()
india_vul=india.loc[india.risk=='Vulnerable',:]

india_endanger=india.loc[india.risk=='Definitely endangered ',:]

india_critical=india.loc[india.risk=='Critically endangered',:]

india_extinct=india.loc[india.risk=='Extinct',:]

india_severe=india.loc[india.risk=='Severely endangered',:]



india_vul.head()


plt.figure(figsize=(15,8))

m = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=40,\

            llcrnrlon=60,urcrnrlon=100,resolution='h')

m.drawparallels(np.arange(30.,60.,5.))

m.drawmeridians(np.arange(60.,100.,10.))

m.drawcoastlines()

m.drawcountries()

colors = ['cyan','green','yellow','orange','red']



x, y = m(list(india_vul["longitude"].astype(float)), list(india_vul["latitude"].astype(float)))

plot1=m.plot(x, y, 'go', markersize = 8, alpha = 0.8, color = colors[0],label='Vulnerable')



x, y = m(list(india_endanger["longitude"].astype(float)), list(india_endanger["latitude"].astype(float)))

plot2=m.plot(x, y, 'go', markersize = 9, alpha = 0.8, color =colors[1],label='Endangered')





x, y = m(list(india_critical["longitude"].astype(float)), list(india_critical["latitude"].astype(float)))

plot3=m.plot(x, y, 'go', markersize = 10, alpha = 0.8, color = colors[2],label='Critical')





x, y = m(list(india_severe["longitude"].astype(float)), list(india_severe["latitude"].astype(float)))

plot4=m.plot(x, y, 'go', markersize = 11, alpha = 0.8, color = colors[3],label='Severe')





x, y = m(list(india_extinct["longitude"].astype(float)), list(india_extinct["latitude"].astype(float)))

plot5=m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = colors[4],label='Extinct')





plt.title('Indian Languages Analysis')

plt.legend()

plt.show()

subcontinent_vul=subcontinent.loc[subcontinent.risk=='Vulnerable',:]

subcontinent_endanger=subcontinent.loc[subcontinent.risk=='Definitely endangered ',:]

subcontinent_critical=subcontinent.loc[subcontinent.risk=='Critically endangered',:]

subcontinent_extinct=subcontinent.loc[subcontinent.risk=='Extinct',:]

subcontinent_severe=subcontinent.loc[subcontinent.risk=='Severely endangered',:]

subcontinent_vul.head()
from pylab import rcParams



rcParams['figure.figsize'] = 30, 30

plt.figure(figsize=(15,8))

m = Basemap(projection='mill',llcrnrlat=-5,urcrnrlat=50,\

            llcrnrlon=65,urcrnrlon=110,resolution='h')



m.drawcoastlines()

m.drawcountries()

colors = ['cyan','green','yellow','orange','red']



x, y = m(list(subcontinent_vul["longitude"].astype(float)), list(subcontinent_vul["latitude"].astype(float)))

plot1=m.plot(x, y, 'go', markersize = 8, alpha = 0.8, color = colors[0],label='Vulnerable')





x, y = m(list(subcontinent_endanger["longitude"].astype(float)), list(subcontinent_endanger["latitude"].astype(float)))

plot2=m.plot(x, y, 'go', markersize = 8, alpha = 0.8, color = colors[1],label='Definitely endangered')



x, y = m(list(subcontinent_critical["longitude"].astype(float)), list(subcontinent_critical["latitude"].astype(float)))

plot3=m.plot(x, y, 'go', markersize = 8, alpha = 0.8, color = colors[2],label='Critically endangered')



x, y = m(list(subcontinent_extinct["longitude"].astype(float)), list(subcontinent_extinct["latitude"].astype(float)))

plot4=m.plot(x, y, 'go', markersize = 8, alpha = 0.8, color = colors[3],label='Extinct')



x, y = m(list(subcontinent_severe["longitude"].astype(float)), list(subcontinent_severe["latitude"].astype(float)))

plot1=m.plot(x, y, 'go', markersize = 8, alpha = 0.8, color = colors[4],label='Severely endangered')





plt.title('Subcontinent Languages Analysis')

plt.legend()



plt.show()