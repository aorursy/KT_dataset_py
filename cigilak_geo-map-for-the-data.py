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
train_df = pd.read_json("../input/train.json")

test_df = pd.read_json("../input/test.json")

train_test = pd.concat([train_df, test_df], 0)

longitude = train_df['longitude'].value_counts()

latitude = train_df['longitude'].value_counts()

map = Basemap(llcrnrlon=-75,llcrnrlat=40,urcrnrlon=-72,urcrnrlat=41, resolution='h',projection='cass',lon_0=-73,lat_0=40)





map.drawmapboundary(fill_color='aqua')

map.fillcontinents(color='coral',lake_color='aqua')

map.drawcoastlines()

lons = np.array(train_df['longitude'])

lats = np.array(train_df['latitude'])

x,y = map(lons, lats)

print(x,y)

map.scatter(x, y, marker='D',color='m')



plt.show()
#Seems Like we have listings that are close to the water bodies. 

#It seems that we have listing in the water too. They are the outliers in the data that we see. 

#Hope that this visualisation would be helpful.