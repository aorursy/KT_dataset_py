import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# clustering algorithms
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabaz_score
data = pd.read_csv('../input/en.openfoodfacts.org.products.tsv', sep='\t')
# get rid of all foods with nan or number as a product name - they are no fun :(
rows_to_delete = []
for i, name in enumerate(data['product_name']):
    if pd.isnull(name) or name.isdigit():
        rows_to_delete.append(i)
data.drop(rows_to_delete, inplace=True)

data.set_index('product_name', inplace=True) # make the index be the food names
# minimizing the number of features we are dealing with in a convinient way:
# get rid of categorical data
data = data.select_dtypes('float64')

for i in data:
    if data[i].isnull().sum() > data.shape[0]*(4/5): # get rid of features which are less than 4/5 filled
        data.drop([i], axis=1, inplace=True)
    elif math.isnan(data[i].mean()):   # get rid of data with no mean
        data.drop([i], axis=1, inplace=True)

data.describe()
# impute the data
my_imputer = SimpleImputer()
updated_data = pd.DataFrame(data = my_imputer.fit_transform(data), index=data.index)

updated_data = (updated_data - np.mean(updated_data)) /(updated_data.max() - updated_data.min())

updated_data.describe()
cluster_count_candidates = np.arange(2, 25)

db_score_list = []
ch_score_list = []
for count in cluster_count_candidates:
    clustering_model = KMeans(n_clusters=count).fit(updated_data)
    db_score_list.append(davies_bouldin_score(updated_data, clustering_model.labels_))
    ch_score_list.append(calinski_harabaz_score(updated_data, clustering_model.labels_))

plt.figure(figsize=(18, 5))

plt.subplot(1,2,1)
plt.stem(cluster_count_candidates, db_score_list)
plt.xlabel('Cluster Count')
plt.ylabel('Davies Bouldin Score')
plt.title('KMeans Model Evaluation (Lower is Better)')

plt.subplot(1,2,2)
plt.stem(cluster_count_candidates, ch_score_list)
plt.xlabel('Cluster Count')
plt.ylabel('Calinski Harabaz Score')
plt.title('KMeans Model Evaluation (Higher is Better)')

plt.show()
cluster_count = 4
k_means_model = KMeans(n_clusters=cluster_count)
clustered_data = k_means_model.fit_predict(updated_data)

# seperate the foods indo cluster lists
clusters = [[] for _ in range(cluster_count)]
for i, cluster in enumerate(clustered_data):
    clusters[cluster].append(data.index[i])
    
# Pad all clusters with 0s to equal the length of the maximum cluser,
# so that we can arrange them all into one dataframe.
max_count = 0
for cluster in clusters:
    if len(cluster) > max_count:
        max_count= len(cluster)

for i in range(cluster_count):
    clusters[i] += [0]*(max_count - len(clusters[i]))\
    
# upload results to output file
results = pd.DataFrame({'Cluster '+str(i+1): clusters[i] for i in range(cluster_count)})
results.to_csv('KMeans_results.csv')
print('Full results can be seen in the output file of this kernel.')
results.head(20)