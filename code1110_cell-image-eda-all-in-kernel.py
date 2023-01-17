# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import sys



import warnings

warnings.filterwarnings("ignore")



import cv2

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.cm import get_cmap

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split



from scipy.stats import norm



import keras

from keras import layers

from keras.models import Model

from keras import metrics

from keras import backend as K   # 'generic' backend so 
train_control = pd.read_csv('../input/train_controls.csv')

test_control = pd.read_csv('../input/test_controls.csv')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

pixel_stats = pd.read_csv('../input/pixel_stats.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
dfs = [train_control, test_control, train, test, pixel_stats, sample_submission]

names = ['train control', 'test control', 'train', 'test', 'pixel stats', 'sample submission']



# display info about a DataFrame

def dispDF(df, name):

    print("========== " + name + " ==========")

    print("SHAPE ----------------------")

    print(df.shape)

    print('')

    print("HEAD ----------------------")

    print(df.head(7))

    print('')

    print("DATA TYPE ----------------")

    print(df.dtypes)

    print('')

    print("NAN counts ----------------")

    print(df.isna().sum())

    print('')

#     print("UNIQUES -------------------")

#     print(df.nunique())

#     print('')

    print("======================================")

    

pd.set_option('display.expand_frame_repr', False)

for df, name in zip(dfs, names):

    dispDF(df, name)
# !git clone https://github.com/recursionpharma/rxrx1-utils

# print ('rxrx1-utils cloned!')
# !ls
# sys.path.append('rxrx1-utils')

# import rxrx.io as rio
# # randomly plot N (RGB) images

# N = 15

# np.random.seed(1220)

# r = np.random.choice(train.shape[0], N)



# fig, ax = plt.subplots(int(N/5), 5, figsize=(24, 18))

# ax = ax.flatten()

# for i in range(N):

#     t = rio.load_site_as_rgb('train', train.loc[r[i], 'experiment'],

#                              train.loc[r[i], 'plate'], 

#                              train.loc[r[i], 'well'], 1)

#     ax[i].imshow(t)

#     ax[i].axis('off')

#     ax[i].set_title(train.loc[r[i], 'id_code'])

# plt.tight_layout()
# image loader

def image_loader(train, row, site, channel, npix):

    experiment = train.loc[row, 'experiment']

    plate = train.loc[row, 'plate']

    well = train.loc[row, 'well']

    img = cv2.imread('../input/train/' + experiment +

                    '/Plate' + str(plate) + '/' + well +

                    '_s' + str(site) + '_w' + str(channel) + '.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    img = cv2.resize(img, (npix, npix)) # resize to npix x npix (for now)

    return img
# randomly plot N (RGB) images

N = 30

np.random.seed(1220)

r = np.random.choice(train.shape[0], N)



fig, ax = plt.subplots(int(N/5), 5, figsize=(24, 18))

ax = ax.flatten()

for i in range(N):

    img = image_loader(train, r[i], np.random.randint(1, 2), 

                       np.random.randint(1, 6), 256)

    ax[i].imshow(img)

    ax[i].axis('off')

    ax[i].set_title(train.loc[r[i], 'id_code'])

plt.tight_layout()
fig, ax = plt.subplots(1, 3, figsize=(12, 8))

ax = ax.flatten()



sns.distplot(train['sirna'], kde=False, norm_hist=False, color='k', ax=ax[0])

ax[0].set_title('train')

sns.distplot(train_control['sirna'], kde=False, norm_hist=False, color='k', ax=ax[1])

ax[1].set_title('train_control')

sns.distplot(test_control['sirna'], kde=False, norm_hist=False, color='k', ax=ax[2])

ax[2].set_title('test_control')
# mean, std, median, min, max for each site and channel

stats = np.zeros((train.shape[0], 60))

columns = ['mean', 'std', 'median', 'min', 'max']

for i, d in enumerate(train['id_code']):

    temp_stats = pixel_stats.loc[pixel_stats['id_code'] == d, columns]

    stats[i, :] = np.reshape(temp_stats.values, (60, ))



print("stats shape: " + str(np.shape(stats)))
# use a subset of the data

uniques = np.unique(train['sirna'].values)

trainX, valX, trainy, valy = train_test_split(stats, train['sirna'].values, test_size=0.9, random_state=1220)



X_decomposed = PCA(n_components=2).fit_transform(trainX)



fig, ax = plt.subplots(1, 1, figsize=(12, 8))

cmap = sns.color_palette("husl", len(uniques))



for i, u in enumerate(uniques):

    marker = "$" + str(u) + "$"

    idx = trainy == u

    ax.scatter(X_decomposed[idx, 0], X_decomposed[idx, 1],

              marker=marker, color=cmap[i])

ax.set_title("PCA")
X_decomposed = TSNE(n_components=2).fit_transform(trainX)



fig, ax = plt.subplots(1, 1, figsize=(12, 8))



for i, u in enumerate(uniques):

    marker = "$" + str(u) + "$"

    idx = trainy == u

    ax.scatter(X_decomposed[idx, 0], X_decomposed[idx, 1],

              marker=marker, color=cmap[i])

ax.set_title("TSNE")