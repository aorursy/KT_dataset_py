import numpy as np 
import pandas as pd 

from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/recomend-music/scrobbler-small-sample.csv')
df.head()
artists_df=pd.read_csv('../input/recomend-music/artists.csv', header=None, names=['artist_name'])
artists_df.head()
artist_names=artists_df['artist_name']
artists_df['artist_offset'] = artists_df.index
artists_df.head()
temp_df=pd.merge(artists_df,df,on="artist_offset")
temp_df.head()
artists=temp_df.pivot_table(columns = 'user_offset',index = 'artist_name',values='playcount',fill_value=0)
artists.head()
# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler ,nmf ,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=artist_names)
df.head()
# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())