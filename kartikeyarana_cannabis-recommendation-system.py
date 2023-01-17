# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/cannabis-strains/cannabis.csv")
data.head()
data.describe()
print(data["Flavor"].isnull().values.sum())
data["Flavor"] = data["Flavor"].fillna("None")
total_flavor = []
max_len = 0
total_flavor.append("None")
for val in data["Flavor"].values:
    print(val)
    val = val.split(",")
    
    if len(val) > max_len:
        max_len = len(val)
    
    for f in val:
        if f not in total_flavor:
            total_flavor.append(f)
            
print(len(total_flavor))
print(max_len)
print(data["Effects"].isnull().values.sum())
# data["Flavor"] = data["Flavor"].fillna("None")
total_effects = []
max_len = 0
for val in data["Effects"].values:
    print(val)
    val = val.split(",")
    
    if len(val) > max_len:
        max_len = len(val)
        
    for e in val:
        if e not in total_effects:
            total_effects.append(e)
print(len(total_effects))
print(max_len)
split_data = data["Flavor"].str.split(",")
split_data = split_data.to_list()
names = ["Flavor_1", "Flavor_2", "Flavor_3", "Flavor_4"]
new_flavor = pd.DataFrame(split_data, columns=names)
new_flavor = new_flavor.fillna("None")
new_flavor["Flavor_4"].isnull().values.sum()

# 2308 values of Flavor_4 column are Null, therefore dropping it.
new_flavor = new_flavor.drop(columns=["Flavor_4"])

new_flavor
split_data = data["Effects"].str.split(",").to_list()
names = ["Effect_1", "Effect_2", "Effect_3", "Effect_4", "Effect_5"]

new_effect = pd.DataFrame(split_data, columns=names).fillna("None")
new_effect
new_effect.describe()
from sklearn.feature_extraction import FeatureHasher
effect_hasher = FeatureHasher(n_features=3, input_type="string")
x = effect_hasher.fit_transform(total_effects)
print(total_effects)
print(x.toarray())
len(np.unique(x.toarray(), axis=0))
flavor_hasher = FeatureHasher(n_features=10, input_type="string")
y = flavor_hasher.fit_transform(total_flavor)
print(total_flavor)
print(y.toarray())
len(np.unique(y.toarray(), axis=0))
total_effects
names = ["Effect_1", "Effect_2", "Effect_3", "Effect_4", "Effect_5"]
num_effect = pd.DataFrame([])
for name in names:
    d = new_effect[name]
    temp = effect_hasher.transform(d).toarray()
    temp = pd.DataFrame(temp)
    num_effect = pd.concat([num_effect, temp], axis=1)
    
print(num_effect.head(3))
print(new_effect.head(3))
names = ["Flavor_1", "Flavor_2", "Flavor_3"]
num_flavor = pd.DataFrame([])
for name in names:
    d = new_flavor[name]
    temp = flavor_hasher.transform(d).toarray()
    temp = pd.DataFrame(temp)
    num_flavor = pd.concat([num_flavor, temp], axis=1)
num_flavor.head(3)
# print(new_flavor.head(1))
new_data = data.copy()
new_data = pd.get_dummies(new_data, columns=["Type"])
new_data
new_data = new_data.drop(columns=["Strain", "Effects", "Flavor", "Description"])
new_data = pd.concat([new_data, num_effect, num_flavor], axis=1)
new_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
# Suppose a customer smoked the 100-Og cannabis and like it. So, I took the features of 100-Og and predicted the recommendations.
test_values = [ 4.,  1.,  0.,  0., -3.,  3.,  0., -2.,  3., -2., -2.,  0.,  0.,
       -1.,  3., -2.,  1.,  3.,  3.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
        1.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,
        0., -2.,  1.,  0.,  0.,  1.,  0.,  0.,  0., -2.]
kmeans = KMeans(n_jobs = -1, n_clusters = 30, init='k-means++', max_iter=500)
model = kmeans.fit(new_data)
test_values = np.array(test_values)
test_values = np.reshape(test_values, (1, -1))
model.predict(test_values)
# KMeans Clustering
# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,100), 'Dist':Dist})
plt.figure(figsize=(50,25))
plt.plot(frame['Cluster'], frame['Dist'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(new_data)
dist, recommends = nbrs.kneighbors([test_values])
print(dist)
print(recommends)
for index in recommends[0]:
    print(data.values[index])
from sklearn.metrics.pairwise import cosine_similarity
res = cosine_similarity(new_data)
print(res)
