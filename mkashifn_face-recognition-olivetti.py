# Imports

import os

import time



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf

from tensorflow.python.client import device_lib

import seaborn as sns

import matplotlib

import matplotlib.gridspec as gridspec

from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from matplotlib.cbook import get_sample_data



from keras.layers import Input, Dense

from keras import regularizers, Model

from keras.models import Sequential



from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, precision_recall_curve, classification_report, confusion_matrix, average_precision_score, roc_curve, auc

from sklearn.datasets import make_blobs

from sklearn.decomposition import PCA

from sklearn import svm

from sklearn import metrics



from imageio import imwrite

#from scipy.misc import imsave



%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from PIL import Image, ImageChops
faces=np.load("/kaggle/input/olivetti/olivetti_faces.npy")

labels=np.load("/kaggle/input/olivetti/olivetti_faces_target.npy")
def print_summary(data, labels):

    print("There are {} images in the dataset".format(len(data)))

    print("There are {} unique labels in the dataset".format(len(np.unique(labels))))

    print("Size of each image is {}x{}".format(data.shape[1],data.shape[2]))

    print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4]))
print_summary(faces, labels)
def show_distinct_faces(images, labels):

    #Creating 4X10 subplots in  18x9 figure size

    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))

    #For easy iteration flattened 4X10 subplots matrix to 40 array

    axarr=axarr.flatten()

    

    #iterating over user ids

    i = 0

    for unique_id in labels:

        image_index=unique_id*10

        axarr[unique_id].imshow(images[image_index], cmap='gray')

        axarr[unique_id].set_xticks([])

        axarr[unique_id].set_yticks([])

        axarr[unique_id].set_title("face id:{}".format(unique_id))

        i += 1

    plt.suptitle(f"There are {i} distinct people in the dataset")
show_distinct_faces(faces, np.unique(labels))
def show_different_faces_of_the_same_person(faces, count, labels):

    cols=count

    rows=(len(labels)*cols)/cols #

    rows=int(rows)

    

    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(18,9))

    #axarr=axarr.flatten()

    

    for i, person_label in enumerate(labels):

        for j in range(cols):

            image_index=person_label*cols + j

            axarr[i,j].imshow(faces[image_index], cmap="gray")

            axarr[i,j].set_xticks([])

            axarr[i,j].set_yticks([])

            axarr[i,j].set_title("person label:{}".format(person_label))
show_different_faces_of_the_same_person(faces, 10, [5, 8, 0, 24])
# Reshape the images for machine learning

X=faces.reshape((faces.shape[0],faces.shape[1]*faces.shape[2]))

print("X shape:",X.shape)
# Split the data into training and testing

X_train, X_test, y_train, y_test=train_test_split(X, labels, test_size=0.3, stratify=labels, random_state=0)

print("X_train shape:",X_train.shape)

print("y_train shape:{}".format(y_train.shape))
def show_2_component_pca(X, labels, count):

    # Pricipal Component Analysis

    pca=PCA(n_components=2)

    pca.fit(X)

    X_pca=pca.transform(X)

    number_of_people=count

    index_range=number_of_people*count

    fig=plt.figure(figsize=(10,8))

    ax=fig.add_subplot(1,1,1)

    scatter=ax.scatter(X_pca[:index_range,0],

                X_pca[:index_range,1], 

                c=labels[:index_range],

                s=25,

               cmap=plt.get_cmap('jet', number_of_people)

              )



    ax.set_xlabel("First Principle Component")

    ax.set_ylabel("Second Principle Component")

    ax.set_title("Two-component PCA projection for {} people".format(number_of_people))



    fig.colorbar(scatter)
show_2_component_pca(X, labels, len(np.unique(labels)))
# Generate a model using 90 components

pca=PCA(n_components=90, whiten=True)

pca.fit(X_train)
X_train_pca=pca.transform(X_train)

X_test_pca=pca.transform(X_test)

clf = svm.SVC()

clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)

print("accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))

print(metrics.classification_report(y_test, y_pred))