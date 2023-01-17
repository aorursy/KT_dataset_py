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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.info()
df.plot(x="longitude", y="latitude", style=".", figsize=(20, 20))
plt.title("Map")
plt.ylabel("latitude")
img = plt.imread("/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png", 0)
plt.imshow(img, extent=[-74.25, -73.685, 40.49, 40.925])
plt.show()
int_col = ['id','host_id','latitude','longitude','price','minimum_nights','number_of_reviews' ,'calculated_host_listings_count','availability_365']
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import sklearn.datasets
dendrogram = sch.dendrogram(sch.linkage(df[int_col][:1500], method='centroid'))
plt.title('dendrogram')
plt.xlabel('price')
plt.ylabel('availability')
plt.show()
