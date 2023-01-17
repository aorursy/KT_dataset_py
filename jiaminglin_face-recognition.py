# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualization

import matplotlib.pyplot as plt



#Machine Learning

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=np.load("../input/olivetti_faces.npy")

target=np.load("../input/olivetti_faces_target.npy")
print("There are {} images in the dataset".format(len(data)))

print("There are {} unique targets in the dataset".format(len(np.unique(target))))

print("Size of each image is {}x{}".format(data.shape[1],data.shape[2]))

print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4]))
print("unique target number:",np.unique(target))
def show_40_distinct_people(images, unique_ids):

    #Creating 4X10 subplots in  18x9 figure size

    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))

    #For easy iteration flattened 4X10 subplots matrix to 40 array

    axarr=axarr.flatten()

    

    #iterating over user ids

    for unique_id in unique_ids:

        image_index=unique_id*10

        axarr[unique_id].imshow(images[image_index], cmap='gray')

        axarr[unique_id].set_xticks([])

        axarr[unique_id].set_yticks([])

        axarr[unique_id].set_title("face id:{}".format(unique_id))

    plt.suptitle("There are 40 distinct people in the dataset")
show_40_distinct_people(data, np.unique(target))
# MODIFICATION, add cols to specify how many pictures displayed for a person

def show_10_faces_of_n_subject(images, subject_ids, cols):



    rows=len(subject_ids) #

    rows=int(rows)

    

    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(18,9))

    #axarr=axarr.flatten()

    

    for i, subject_id in enumerate(subject_ids):

        for j in range(cols):

            image_index=subject_id*cols + j

            axarr[i,j].imshow(images[image_index], cmap="gray")

            axarr[i,j].set_xticks([])

            axarr[i,j].set_yticks([])

            axarr[i,j].set_title("face id:{}".format(subject_id))
#You can playaround subject_ids to see other people faces

show_10_faces_of_n_subject(images=data, subject_ids=[0,5, 21, 34, 36], cols = 10)
#We reshape images for machine learnig  model

X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))

print("X shape:",X.shape)
X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)

print("X_train shape:",X_train.shape)

print("y_train shape:{}".format(y_train.shape))
show_10_faces_of_n_subject(images=X_train.reshape(280, 64,64), subject_ids=[0,5, 21, 34, 36], cols = 6)
# number of labels in a categoriest

y_frame=pd.DataFrame()

y_frame['subject ids']=y_train

y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,8),title="Number of Samples for Each Classes")