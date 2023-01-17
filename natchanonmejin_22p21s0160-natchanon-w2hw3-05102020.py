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
airb= pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airb.count()
airb.isnull().sum()
airb["host_name"].fillna(method ='ffill', inplace = True)
airb.isnull().sum()
airb["name"].fillna(method ='ffill', inplace = True)
airb.isnull().sum()
airb= airb.drop(['last_review','reviews_per_month'], axis=1)
airb.isnull().sum()
import seaborn as sns
sns.lmplot(data=airb, x='longitude', y='latitude', hue='neighbourhood_group', 
                   fit_reg=False, legend=True, legend_out=True)
sns.lmplot(data=airb, x='longitude', y='latitude', hue='room_type', 
                   fit_reg=False, legend=True, legend_out=True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

airb.name= le.fit_transform(airb['name'].values)
airb.host_name= le.fit_transform(airb['host_name'].values)
airb.neighbourhood_group= le.fit_transform(airb['neighbourhood_group'].values)
airb.neighbourhood= le.fit_transform(airb['neighbourhood'].values)
airb.room_type= le.fit_transform(airb['room_type'].values)
hie= airb.sample(4000)
hie
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize = (30, 3))
plt.title("Airbnb New York 2019 Dendograms")
dend= shc.dendrogram(shc.linkage(hie, method = 'ward'))
