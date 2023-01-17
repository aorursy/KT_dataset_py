import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
print(data.isnull().sum())
print(data.head())
#convert label to int
data.neighbourhood_group=data.neighbourhood_group.astype('category').cat.codes
data.neighbourhood=data.neighbourhood.astype('category').cat.codes
data.room_type=data.room_type.astype('category').cat.codes
print(data.head())
# input data
# features = data[["neighbourhood_group", "neighbourhood", "latitude", "longitude", "room_type", "price", "minimum_nights", "number_of_reviews", "availability_365"]]
features = np.array(data[["neighbourhood_group", "price"]])
print(features[5000:10000,:].shape)
del data
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
predictions = cluster.fit_predict(features[5000:10000,:])
print(np.unique(predicts))
predictions = predictions.reshape(-1,1)
inputs = features[5000:10000,:]
print(predictions.shape)
print(inputs.shape)
output = np.concatenate((inputs, predictions),axis=1)
print(output.shape)
plt.figure(figsize=(10, 7))
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(output, method='ward'))