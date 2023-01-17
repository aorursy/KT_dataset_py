# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import pyarrow.parquet as pq

import matplotlib.pyplot as plt





# Any results you write to the current directory are saved as output.
def random_crop(data,meta_data = None,length = 50000,n_samples = 100,targets = None,mode = 'uniform',sigma = 400000):

    if meta_data is None:

        meta_data = pd.read_csv('../input/metadata_train.csv')

    if length is None:

        length = np.random.randint(1,len(data))

    if mode != 'uniform':

        start = np.random.randn() * sigma + len(data) // 2

    else:

        start = np.random.randint(1,len(data))

    if isinstance(targets,int):

        meta_data = meta_data[meta_data['target'] == targets]

    indices = range(len(meta_data))

    #selected_data = data[:,indices]

    if n_samples > meta_data.shape[0]:

        n_samples = meta_data.shape[0]

    inds = np.random.choice(indices,n_samples,replace = False)

    targets = meta_data.iloc[inds].target.values

    pair = (data[start:start + length,inds],targets)

    return  pair

     
# assumes random crop is done

def synthesize_waves(pair_a,pair_b,alpha = .5):

    assert np.all(pair_a[1] == pair_b[1]), 'class does not match'

    assert len(pair_a[0]) == len(pair_b[0]), 'wave length does not match'

    np.random.shuffle(pair_b[0].T)

    return alpha * pair_a[0] + (1 - alpha) * pair_b[0]

#test and show some results here

data  = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(510)]).to_pandas().values

meta_data = pd.read_csv('../input/metadata_train.csv')[:510]
pair_a = random_crop(data,meta_data) 
plt.plot(pair_a[0][:,3])
pair_b = random_crop(data,meta_data)
plt.plot(pair_b[0][:,3])
#synthesize the waves

pair_a = random_crop(data,meta_data,targets = 0)

pair_b = random_crop(data,meta_data,targets = 0)

synthesized = synthesize_waves(pair_a,pair_b)

plt.plot(synthesized[:,3])