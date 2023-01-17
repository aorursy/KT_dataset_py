%load_ext autoreload

%autoreload 2
from sklearn.manifold import MDS
import pandas as pd

import numpy as np

import pathlib

from itertools import islice

import tensorflow as tf

from tqdm import tqdm

from tqdm.keras import TqdmCallback

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error 

from sklearn.model_selection import KFold

from sklearn.svm import LinearSVR

from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from sklearn.svm import SVR



import matplotlib.pyplot as plt



%matplotlib inline
data = pathlib.Path('/kaggle/input/trends-assessment-prediction')
df_fnc = pd.read_csv(data / 'fnc.csv')



df_reveal_ID = pd.read_csv(data / 'reveal_ID_site2.csv')



df_train = pd.read_csv(data / 'train_scores.csv')



df_test = pd.read_csv(data / 'sample_submission.csv')



df_loading = pd.read_csv(data / 'loading.csv')
targets = list(df_train.columns)

targets.remove('Id')
corr = df_train[targets].corr()
corr
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
trans = mds.fit_transform(1 - abs(corr))
trans
x, y = list(zip(*trans))



plt.scatter(x, y)

for x1, y1, text1 in zip(x, y, targets):

    plt.text(x1, y1, text1)



frame1 = plt.gca()

frame1.axes.xaxis.set_ticklabels([])

frame1.axes.yaxis.set_ticklabels([])



plt.xticks([])

plt.yticks([])



plt.xlim(-1, 1)

plt.ylim(-1, 1)



plt.title('"clossness" map based on correlation among targets')

plt.show()
train_merged = df_train.merge(df_loading, on='Id', how='left')



train_merged = train_merged.drop(['Id'], axis=1);
mds_with_loading = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
corr_with_loading = train_merged.corr()



trans_with_loading = mds_with_loading.fit_transform(1 - abs(corr_with_loading))
from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')



x, y = list(zip(*trans_with_loading))



plt.scatter(x, y)

for x1, y1, text1 in zip(x, y, train_merged.columns):

    plt.text(x1, y1, text1)



frame1 = plt.gca()

frame1.axes.xaxis.set_ticklabels([])

frame1.axes.yaxis.set_ticklabels([])



plt.xticks([])

plt.yticks([])



plt.xlim(-1, 1)

plt.ylim(-1, 1)



plt.title('"clossness" map based on correlation among targets & loading')

plt.show()