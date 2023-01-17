# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.cluster import hierarchy
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
df
def onehot(a):
    return OneHotEncoder().fit_transform(np.array(a).reshape(-1, 1))

def build_features(df):
    res = []
    
    res.append(LabelEncoder().fit_transform(df['neighbourhood_group']))
    res.append(LabelEncoder().fit_transform(df['neighbourhood']))
    res.append(LabelEncoder().fit_transform(df['room_type']))
    
    res.append(list(df['latitude']))
    res.append(list(df['longitude']))
    res.append(list(df['price']))
    res.append(list(df['minimum_nights']))
    res.append(list(df['number_of_reviews']))
    res.append(list(df['reviews_per_month'].fillna(0)))
    res.append(list(df['calculated_host_listings_count']))
    res.append(list(df['availability_365']))
    
    return np.array(res).T
features = build_features(df[:50])
features

Z = hierarchy.linkage(features, 'complete')
plt.figure(figsize=(30,10))
dn = hierarchy.dendrogram(Z)
plt.show()
features = build_features(df[:100])
features

Z = hierarchy.linkage(features, 'complete')
plt.figure(figsize=(30,10))
dn = hierarchy.dendrogram(Z)
plt.show()
features = build_features(df[:250])
features

Z = hierarchy.linkage(features, 'complete')
plt.figure(figsize=(40,10))
dn = hierarchy.dendrogram(Z)
plt.show()
features = build_features(df[:500])
features

Z = hierarchy.linkage(features, 'complete')
plt.figure(figsize=(10,75))
dn = hierarchy.dendrogram(Z, orientation='right')
plt.show()
features = build_features(df[:1000])
features

Z = hierarchy.linkage(features, 'complete')
plt.figure(figsize=(10,150))
dn = hierarchy.dendrogram(Z, orientation='right')
plt.show()
