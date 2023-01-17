# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls -al /kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv
data_in = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data_in
data_in.info()
print('len(host_id): ', len(set(data_in.host_id)))
print('len(host_name): ', len(set(data_in.host_name)))
print('neighbourhood_group: ', set(data_in.neighbourhood_group))
print('room_type: ', set((data_in.room_type)))
data = data_in.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','price','availability_365','minimum_nights']]
data
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

data_label = data['neighbourhood'].to_list()
data_val = [[x] for x in data['price']]

plt.figure(figsize=(30, 60), facecolor='white')
plt.title("neighbourhood vs price")
dend = shc.dendrogram(shc.linkage(data_val, method='complete'), labels=data_label, orientation="left")
plt.show()
data_val = [[x] for x in data['availability_365']]

plt.figure(figsize=(30, 60), facecolor='white')
plt.title("neighbourhood vs availability_365")
dend = shc.dendrogram(shc.linkage(data_val, method='complete'), labels=data_label, orientation="left")
plt.show()
data_val = [[x] for x in data['minimum_nights']]

plt.figure(figsize=(30, 50), facecolor='white')
plt.title("neighbourhood vs minimum_nights")
dend = shc.dendrogram(shc.linkage(data_val, method='complete'), labels=data_label, orientation="left")
plt.show()
X = list(zip(data.loc[:,['availability_365']]['availability_365'].tolist(), data.loc[:,['price']]['price'].tolist()))

plt.figure(figsize=(30, 30), facecolor='white')
plt.scatter(data.loc[:,['availability_365']].values, data.loc[:,['price']].values, label='availability_365 vs price')

for label, x, y in zip(data_label, data.loc[:,['availability_365']].values, data.loc[:,['price']].values):
    plt.annotate(label, xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
plt.title("neighbourhood vs availability_365 vs price")
plt.xlabel('availability_365')
plt.ylabel('price')
plt.show()
plt.figure(figsize=(30, 50), facecolor='white')
plt.title("neighbourhood vs availability_365 vs price")
dend = shc.dendrogram(shc.linkage(X, method='complete'), labels=data_label, orientation="left")
plt.show()