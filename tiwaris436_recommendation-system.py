import pandas as pd 
data = pd.read_csv('../input/scrobbler-small-sample.csv')

import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from scipy import sparse
from scipy.sparse import csr_matrix
print(data.head())
print(data.shape)
max(data['artist_offset'])
max(data['user_offset'])
artists1 = data.sort_values(['artist_offset', 'user_offset'], ascending=[True, True])
print(artists1.head())


row_ind = np.array(artists1['artist_offset'])
col_ind = np.array(artists1['user_offset'])
data1 = np.array(artists1['playcount'])
artists = sparse.coo_matrix((data1, (row_ind, col_ind)))
print(artists)
print(artists.shape)
scaler = MaxAbsScaler()
nmf = NMF(n_components=20)
normalizer = Normalizer()
pipeline = make_pipeline(scaler, nmf, normalizer)
norm_features = pipeline.fit_transform(artists)
#print(type(norm_features))
#print(norm_features)
norm_features.shape

artist_names = pd.read_csv('../input/artists.csv',header=None)
#print(type(artist_names))
#print(len(artist_names))
print(artist_names)
    
print(artist_names.shape)
artist_names1 = artist_names.values.tolist()
print(type(artist_names1))
print(len(artist_names1))
#print(artist_names1)


flattened = []
for sublist in artist_names1:
    for val in sublist:
        flattened.append(val)
        
flattened  = [val for sublist in artist_names1 for val in sublist]
flattened 
df = pd.DataFrame(norm_features, index = flattened)
artist = df.loc['Bruce Springsteen']
similarities = df.dot(artist)
print(similarities.nlargest(10))
# If someone listens songs of singer 'Bruce Springsteen' then he or she will also like songs by below listed Singers. 
