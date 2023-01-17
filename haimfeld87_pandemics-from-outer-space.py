# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
meteor = pd.read_csv('../input/meteorite-landings/meteorite-landings.csv')



meteor = meteor[meteor.year != 2501]

meteor = meteor[meteor.year != 2101]
meteor
from mpl_toolkits.basemap import Basemap

# Extract the data we're interested in

lat = meteor['reclat'].values

lon = meteor['reclong'].values
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

fig = plt.figure(figsize=(12,10))



x,y = m(lon,lat)



plt.title("Locations of meteors (red)")

m.plot(x, y, "o", markersize = 5, color = 'red')



m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
from pandas import Series

def fig_p(data):

    series=Series(data).value_counts().sort_index()

    series.plot(kind='bar')
year = sorted(i for i in meteor['year'] if i >= 2000)



plt.figure(figsize=(10,5))

fig_p(year)

plt.ylabel("Count")

plt.title("Number of meteors per year (>2000)")
import requests

from bs4 import BeautifulSoup 

urlreq = requests.get('https://en.wikipedia.org/wiki/List_of_epidemics')

if urlreq.status_code == 200:

    print("Successful request")

else:

    print(urlreq.status_code)

url = urlreq.text

soup = BeautifulSoup(url, 'lxml')



all_tables = soup.find_all('table')

my_table = soup.find('table', {'class':'wikitable sortable'})



B1 = []

B2 = []

B3 = []

B4 = []

B5 = []

B6 = []



for row2 in my_table.find_all('tr'):

    cells2 = row2.find_all('td')

    if len(cells2)== 6:

        B1.append(cells2[0].find(text = True))

        B2.append(cells2[1].find(text = True))

        B3.append(cells2[2].find(text = True))

        B4.append(cells2[3].find(text = True))

        B5.append(cells2[4].find(text = True))

        B6.append(cells2[5].find(text = True))



zippedlist1 = list(zip(B1, B2, B3, B4, B5, B6))

epidemic = pd.DataFrame(zippedlist1, columns=['Event', 'Date', 'Location','disease','Death toll','Ref'])

epidemic
year2 = []



for i in (epidemic['Date'].values.tolist()):

    year2.append(i[0:4])



year2 = ([s.strip('–') for s in year2])

year2 = ([s.strip('\n') for s in year2])

year2 = list(map(int, year2))



year2 = sorted(i for i in year2 if i >= 2000 and i <=2013)





plt.figure(figsize=(10,5))

fig_p(year2)

plt.ylabel("Count")

plt.title("Number of epidemics per year (>2000)")
series=Series(year).value_counts(normalize=True).sort_index()

series2=Series(year2).value_counts(normalize=True).sort_index()[0:14]





x = np.array(list(range(2000, 2014, 1)))



plt.figure(figsize=(10,5))

plt.bar(x + 0.00, series, color = 'b', width = 0.25)

plt.bar(x + 0.25, series2, color = 'g', width = 0.25)

plt.legend(('meteors', 'epidemics'))

plt.ylabel("Count")

plt.xlabel("Year")

plt.title("Normalized number of epidemics and meteors per year (>2000)")

plt.show()