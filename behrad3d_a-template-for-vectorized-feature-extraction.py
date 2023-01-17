import os

import time

import pandas as pd

import numpy as np

from tqdm import tqdm_notebook
%%time

train_df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}) 
# Segmenting 

rows = 150_000

segments = int(np.floor(train_df.shape[0] / rows))

print("Number of segments: ", segments)
def create_features(seg_id, seg, X):

    xc = pd.Series(seg['acoustic_data'].values)

    

    X.loc[seg_id, 'mean'] = xc.mean()

    X.loc[seg_id, 'std'] = xc.std()

    X.loc[seg_id, 'max'] = xc.max()

    X.loc[seg_id, 'min'] = xc.min()

    X.loc[seg_id, 'sum'] = xc.sum()

    X.loc[seg_id, 'median'] = xc.median()

    

    

    return X
train_X = pd.DataFrame(index=range(segments), dtype=np.float64)

train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
%%time

for seg_id in tqdm_notebook(range(segments)):

    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]

    create_features(seg_id, seg, train_X)

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
# This function makes sure the input matrix is dividable by the target number of rows

def prep_df_for_separation(df,rows):

    mod_value = df.shape[0] % rows 

    if mod_value > 0:

        lastRow = df.shape[0] - mod_value

        df = df.iloc[:lastRow]

    return df 
# an example of vactorized feature exraction function

def vectorized_features(data):

    n_features = 6

    output_matrix = np.empty(shape=(data.shape[1], n_features))



    output_matrix[:,0] = np.mean(data,axis=0)

    output_matrix[:,1] = np.std(data,axis=0)

    output_matrix[:,2] = np.max(data,axis=0)

    output_matrix[:,3] = np.min(data,axis=0)

    output_matrix[:,4] = np.sum(data,axis=0)

    output_matrix[:,5] = np.median(data,axis=0)

    

    return output_matrix
%%time

train_df = prep_df_for_separation(train_df,rows)

data_matrix = train_df.acoustic_data.values.reshape(-1,rows).T

output_matrix_all = train_df.time_to_failure.values.reshape(-1,rows).T

output_matrix = np.min(output_matrix_all,axis=0)



print("data matrix shape", data_matrix.shape)

print("output matrix shape", output_matrix.shape)
%%time

features = vectorized_features(data_matrix)

print("data matrix shape", data_matrix.shape, "\t| Output matrix shape:", output_matrix.shape)