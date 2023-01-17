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
df_target_train = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_target_train.csv')

print('df_target_train:', df_target_train.shape)



df_sample_submit = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_sample_submit.csv')

print('df_sample_submit:', df_sample_submit.shape)



df_tracks = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_tracks.csv')

print('df_tracks:', df_tracks.shape)



df_genres = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_genres.csv')

print('df_genres:', df_genres.shape)



df_features = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_features.csv')

print('df_features:', df_features.shape)

df_target_train.head()
df_genres.head()
df_genres=df_genres.rename(columns={"title":"genre"})



map_name=df_genres.set_index("genre_id")["genre"].to_dict()

map_name[0]="root"

df_genres["parent_name"]=df_genres["parent"].map(map_name)

df_genres.head()
import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline



G = nx.from_pandas_edgelist(df_genres, 'genre', 'parent_name')

fig, ax = plt.subplots(figsize=(20, 20))

nx.draw(G, with_labels=True,ax=ax)

plt.show()