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
%matplotlib inline

from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import complete, fcluster, dendrogram, linkage

import matplotlib.pyplot as plt

import math
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv',)

df.head()
col = ['host_id','neighbourhood_group','latitude','longitude','room_type','price','number_of_reviews','reviews_per_month']

data = df[col]

data = data.fillna(0)

city_list = pd.unique(df.neighbourhood_group).tolist()

room_list = pd.unique(df.room_type).tolist()



data['neighbourhood_group'] = pd.Categorical(data['neighbourhood_group'],categories=city_list)

data['neighbourhood_group'] = data['neighbourhood_group'].cat.codes



data['room_type'] = pd.Categorical(data['room_type'],categories=room_list)

data['room_type'] = data['room_type'].cat.codes





data.head()
samples = data.sample(n=100)

samples = samples.loc[:, col].values

Z = complete(samples)

fig = plt.figure(figsize=(15, 10))

dendrogram(Z)

plt.title('Dendrogram')

plt.xlabel('Rooms / Houses')

plt.ylabel('Euclidean distances')

plt.show()