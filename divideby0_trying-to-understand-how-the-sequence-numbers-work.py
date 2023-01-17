# imports...

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

from scipy.io import loadmat
# Based off earlier mat_to_pandas by ZFTurbo

def mat_to_pandas(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    sequence = -1

    if 'sequence' in names:

        sequence = mat['dataStruct']['sequence'][0,0][0,0]

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]), sequence
df1, seq1 = mat_to_pandas('../input/train_1/1_1_1.mat')

df2, seq2 = mat_to_pandas('../input/train_1/1_2_1.mat')

print(seq1, ',', seq2)
mat1 = df1.as_matrix()

mat2 = df1.as_matrix()

# mat3 is just some nan's so we can see the seperation where the lines should line up

mat3 = mat1[[-1],:] * np.nan 

mat4 = np.r_[mat1[-20:,:], mat3, mat2[:20,:]]

# Plot last 20 samples of sequence 1, and first 20 samples of sequence 2

plt.plot(mat4);