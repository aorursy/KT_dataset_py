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
song_df = pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data.csv")
song_df.head()
data_desc = song_df.describe()
data_desc
data_desc.to_csv("../working/data_desc.csv")
song_filtered = song_df[song_df.popularity >= 30]
song_filtered.count()
features = song_filtered.drop(["id", "artists", "explicit","mode", "popularity", "release_date",  "year", "name", "duration_ms"], axis=1)
features.head()
songs = song_filtered[["id", "name", "artists", "explicit","mode", "popularity",  "year", "duration_ms"]]
songs = songs.rename({"id": "song_id"}, axis="columns")
songs = songs.astype({'mode':'int32', 'explicit':'int32'})
songs.head()
songs.count()
songs.to_csv('../working/songs.csv', index_label='id')
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(15, 15))
sns.despine(left=True)
ft = features.drop("key", axis=1)
for i, col in enumerate(ft.columns):
    r = i // 3
    c = i % 3
    sns.distplot(features[col], color="m", ax=axes[r, c])
sns.distplot(features["valence"], color="m")
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import json
feature_vals = features.values
feature_vals = minmax_scale(feature_vals)
sample_size = 5000
np.random.seed(13437375)
sample_ids = np.random.choice(feature_vals.shape[0], sample_size, replace=False)
feature_samples = feature_vals[sample_ids, :]
scaled_features = pd.DataFrame(feature_vals, columns=features.columns)
scaled_features.to_csv('../working/scaled_features.csv')
songs_temp = songs.reset_index()
embedding = Isomap(n_components=3)
feature_transformed = embedding.fit_transform(feature_samples)
df = pd.DataFrame(feature_transformed, columns=['X', 'Y', 'Z'])
df_isomap = df
df.head()
# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
pca = PCA(n_components=3)
linear_embeddings = pca.fit_transform(feature_samples)
df = pd.DataFrame(linear_embeddings, columns=['X', 'Y', 'Z'])
df_pca = df
df.head()
# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
#idx for 4e5lEqydMatcEio6ef9Dpf (Phenomenal by Eminem)
sample_id = songs_temp[songs_temp.song_id == '4e5lEqydMatcEio6ef9Dpf'].index.tolist()[0]
print(sample_id)
sample_id = np.where(sample_ids == sample_id)[0][0]
print(sample_id)
sample_ids[sample_id]
# cosine
def cosine_dist(u, v):
    u_l2 = np.sqrt(np.dot(np.transpose(u), u))
    v_l2 = np.sqrt(np.dot(np.transpose(v), v))
    
    numerator = np.dot(np.transpose(u), v)
    dist = np.arccos((numerator)/(u_l2*v_l2))/np.pi
    return dist

# euclidean
def euclidean_dist(u, v):
    a = np.subtract(u, v)
    dist = np.dot(np.transpose(a), a)
    return dist
def dist_2songs(feature1, feature2, metric="cosine"):
    print(metric)
    dist = -1;
    if(metric == "cosine"):
        #print("calculating cosine")
        dist = cosine_dist(feature1, feature2)
    elif(metric == "euclidean"):
        #print("calculating euclidean")
        a = np.subtract(feature1, feature2)
        dist = np.dot(np.transpose(a), a)
        
    return dist

distances = []
for i in feature_samples:
    distances.append(cosine_dist(feature_samples[sample_id], i))
    
print(len(distances))
cosine_recoms = np.argsort(np.array(distances))[1:16]
cosine_recoms = sample_ids[cosine_recoms]
cosine_rec_name = songs_temp.name[cosine_recoms].tolist()
cosine_rec_ids = songs_temp.song_id[cosine_recoms].tolist()

print(cosine_rec_name)   
    
distances = []
for i in feature_samples:
    distances.append(euclidean_dist(feature_samples[sample_id], i))
    
print(len(distances))
euc_recoms = np.argsort(np.array(distances))[1:16]
euc_recoms = sample_ids[euc_recoms]
euc_rec_name = songs_temp.name[euc_recoms].tolist()
euc_rec_ids = songs_temp.song_id[euc_recoms].tolist()

print(euc_rec_name)    
    
distances = []
for i in df_pca.values:
    distances.append(euclidean_dist(df_pca.values[sample_id], i))
    
print(len(distances))
pca_recoms = np.argsort(np.array(distances))[1:16]
pca_recoms = sample_ids[pca_recoms]
pca_rec_name = songs_temp.name[pca_recoms].tolist()
pca_rec_ids = songs_temp.song_id[pca_recoms].tolist()

print(pca_rec_name)    
    
distances = []
for i in df_isomap.values:
    distances.append(euclidean_dist(df_isomap.values[sample_id], i))
    
print(len(distances))
iso_recoms = np.argsort(np.array(distances))[1:16]
iso_recoms = sample_ids[iso_recoms]
iso_rec_name = songs_temp.name[iso_recoms].tolist()
iso_rec_ids = songs_temp.song_id[iso_recoms].tolist()

print(iso_rec_name)    
    
df = pd.DataFrame({'cosine':cosine_rec_name, 'euclidean': euc_rec_name, 'pca': pca_rec_name, 'isomap': iso_rec_name})
df
def get_recommendation(current_feature, threshold):
    a = np.subtract(feature_vals, current_feature)
    a = np.multiply(a, a)
    distances = np.sum(a, axis=1)
    distances = distances[distances > threshold]
    recom_list = songs_temp.song_id[np.argsort(distances)[1:4]].tolist()
    return recom_list
def create_recommendations(idx):
    recommendations = []
    for i in range(idx, min(idx+sample_size, feature_vals.shape[0])):
        recommendation = dict()
        recommendation['name'] = songs_temp.song_id[i]
        recommendation["neighbors"] = get_recommendation(feature_vals[i], 0.5)
        recommendations.append(recommendation)
    return recommendations

import time
start_time = time.time()

for i in range(0, feature_vals.shape[0], sample_size):
    recom = create_recommendations(i)
    filename = f"../working/recom_{i}.json"
    print(f"writing file {i/sample_size}")
    with open(filename, "w") as outfile:
        json.dump(recom, outfile) 

print("--- %s seconds ---" % (time.time() - start_time))

dist_2songs('3rJAP9G6hjx2kN8Jsvas09', '1fipvP2zmef6vN2IwXfJhY')
