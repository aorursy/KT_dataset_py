# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
episode_name_list = os.listdir('/kaggle/input/chai-time-data-science/Cleaned Subtitles')
def preprocess(df):
    df = df[df['Speaker']!='Sanyam Bhutani']['Text']
    return df
results  = {}
for episode in episode_name_list:
    df = pd.read_csv("/kaggle/input/chai-time-data-science/Cleaned Subtitles/"+episode)
    text = preprocess(df)
    sentence_embeddings = model.encode(text)
    results[episode.replace('.csv','')] = sentence_embeddings
episode_embeddings = {k:np.mean(v,axis=0) for k,v in results.items()}
for k, v in episode_embeddings.items():
    if v.shape == ():
        print(k)
del episode_embeddings['E69']
episode_embeddings_list = list(episode_embeddings.values())
episode_ids = list(episode_embeddings.keys())

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(episode_embeddings_list)
cluster_assignment = clustering_model.labels_
cluster_assignment
clustered_episodes = [[] for i in range(num_clusters)]
for episode_id, cluster_id in enumerate(cluster_assignment):
    clustered_episodes[cluster_id].append(episode_ids[episode_id])

clustered_episodes
episode_names = pd.read_csv("/kaggle/input/chai-time-data-science/Episodes.csv")
episode_descriptions = pd.read_csv("/kaggle/input/chai-time-data-science/Description.csv")
episode_names.head()
episode_mapping = pd.Series(episode_names['episode_name'].values,index=episode_names['episode_id']).to_dict()
clustered_names = []
for index, cluster in enumerate(clustered_episodes):
    print("\n")
    print("Cluster ",index+1)
    print("\n")
    cluster_list = []
    for episode in cluster:
        print(episode_mapping[episode])
        cluster_list.append(episode_mapping[episode])

        clustered_names.append(cluster_list)
