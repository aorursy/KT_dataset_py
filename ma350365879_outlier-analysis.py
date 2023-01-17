# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
# read data
train = pd.read_csv('../input/train.csv')
# get the most frequent place_id
place_id_freq = list(train.place_id.value_counts().index)
print(place_id_freq[:10])
from sklearn.neighbors import NearestNeighbors
import scipy.stats as ss

place = place_id_freq[10] # the id of place id

sample_train = train[train.place_id==place]

nbrs = NearestNeighbors(n_neighbors=2).fit(sample_train[['x','y']])
distances, indices = nbrs.kneighbors(sample_train[['x','y']])

sample_train['avg_dist'] = ss.zscore(np.mean(distances[:,1:],axis=1))
sample_train['is_outlier'] = (sample_train['avg_dist']>10).astype(int)

sample_train.plot(kind='scatter',x='x',y='y',c=sample_train.is_outlier)
plt.title("place_id: "+str(place))
