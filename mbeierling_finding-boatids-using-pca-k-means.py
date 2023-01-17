# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import random

import cv2

import time

import os

from keras.utils import np_utils



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



from sklearn.decomposition import RandomizedPCA

from sklearn.cluster import KMeans
def get_im_cv2(path):

    img = cv2.imread(path)

    resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)

    return resized





def load_train():

    X_train = []

    X_train_id = []

    y_train = []

    start_time = time.time()



    print('Read train images')

    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    for fld in folders:

        index = folders.index(fld)

        print('Load folder {} (Index: {})'.format(fld, index))

        path = os.path.join('..', 'input', 'train', fld, '*.jpg')

        files = glob.glob(path)

        for fl in files:

            flbase = os.path.basename(fl)

            img = get_im_cv2(fl)

            X_train.append(img)

            X_train_id.append(fl)

            y_train.append(index)



    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return X_train, y_train, X_train_id





def read_and_normalize_train_data():

    train_data, train_target, train_id = load_train()



    print('Convert to numpy...')

    train_data = np.array(train_data, dtype=np.uint8)

    train_target = np.array(train_target, dtype=np.uint8)



    print('Reshape...')

    train_data = train_data.transpose((0, 3, 1, 2))



    print('Convert to float...')

    train_data = train_data.astype('float32')

    train_data = train_data / 255

    train_target = np_utils.to_categorical(train_target, 8)



    print('Train shape:', train_data.shape)

    print(train_data.shape[0], 'train samples')

    return train_data, train_target, train_id
train_x, train_y, train_id = read_and_normalize_train_data()
np_x = np.array(train_x)
data_final = np_x.reshape(3777, 3072)

data_final
n_components = 50

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data_final)
x_train_pca = pca.transform(data_final)
n_boats = 12
kmeans = KMeans(n_clusters=n_boats, random_state=0).fit(x_train_pca)
predicted_labels = kmeans.predict(x_train_pca)
predicted_labels
# checking for redundant image file strings across the fish-folders

len(np.unique(np.array(train_id))) == len(train_x)
# NDA doesn't allow images in public notebook.

# Run this code in local notebook to see the clustering results. Spoiler: clusters look fine ;-)

'''

for cluster in range(0, kmeans.n_clusters):

    cluster_counter = 0

    cluster_predictions = predicted_labels == cluster

    

    _, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(10, 3))

    plt.suptitle("Cluster No.{}".format(cluster + 1, size=20))

                 

    for idx in range(0, len(cluster_predictions)):

        if cluster_predictions[idx]:

            img = mpimg.imread(train_id[idx])

            ax[cluster_counter // 4, cluster_counter % 4].imshow(img)

            cluster_counter += 1

            if (cluster_counter == 8):

                break

                

    plt.show()

'''