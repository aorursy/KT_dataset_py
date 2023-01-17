# In this experiment i'm perform K-Means clustering to try to find

# 'relevant' patterns in objects from coil100 dataset

# the basic idea is that the clusters are equals to classes from object

# or unknown patterns 



# the future applications from this experiment is to find patterns on clinical data 

# (how to discover similar diseases to prevent using unsupervised techniques)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import glob

import cv2

from sklearn.preprocessing import StandardScaler

import random

from matplotlib import rcParams
dataset_path = '../input/coil100/coil-100/coil-100/*.png'

images_filenames = glob.glob(dataset_path)

images_filenames[0]
fig, ax = plt.subplots(1,1)

img = cv2.cvtColor(cv2.imread(images_filenames[0]), cv2.COLOR_BGR2GRAY)

ax.imshow(img, cmap="gray")
def get_clustering (k, data):

    k_means = KMeans(n_clusters=k, init='k-means++', n_init=30, max_iter=20,

                     tol=0.0001, precompute_distances=True, verbose=0,

                     random_state=None, copy_x=True, n_jobs=-1)



    k_means.fit(data)

    return k_means
# variance filter method

def apply_variance_filter(dataset, variance_threshold):

    if variance_threshold != 1:

        cut_point = round(variance_threshold * dataset.shape[1])



        variances = dataset.var()

        data = pd.DataFrame({"variance": variances, "feature": dataset.columns.values})

        data_sorted = data.sort_values(by="variance", ascending=False)



        data_filtered = data_sorted.iloc[0:int(cut_point), :]

        dataset = dataset.loc[:, data_filtered.loc[:, "feature"].values]



    return dataset
def show_labeled_images (model):

    sample = random.choice(np.unique(model.labels_))

    images_index = np.where(model.labels_ == sample)[0]

    

    %matplotlib inline

    

    rcParams['figure.figsize'] = 15 ,15

    

    image_array = []



    for file_index in images_index:

        image_array.append(images_filenames[file_index])



    # check the first 10 images from 1st cluster

    images_to_show = image_array[0:10]

    

    fig, ax = plt.subplots(1,len(images_to_show))



    i = 0

    for img_path in images_to_show:

        # img = mpimg.imread(img_path)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

        ax[i].imshow(img, cmap="gray")

        i = i + 1
dataset = pd.read_csv("../input/coil100-gray-preprocessed/coil_preprocessed.csv", sep=";")



dataset.head()
new_dataset = apply_variance_filter(dataset, 0.25)



new_dataset_standardized = StandardScaler().fit_transform(new_dataset)

pd.DataFrame(new_dataset_standardized).head()
# TODO: search for k clusters instead of use magic number 10

model1 = get_clustering(10, new_dataset_standardized)
show_labeled_images(model1)
dataset_standardized = StandardScaler().fit_transform(dataset)

pd.DataFrame(dataset_standardized).head()
model2 = get_clustering(10, dataset_standardized)
show_labeled_images(model2)