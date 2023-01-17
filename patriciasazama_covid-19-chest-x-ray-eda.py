# This code is almost a direct clone of the malaria EDA found in my other notebook. CNN portion was not used due to small dataset size.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



train_normal_files = []

train_pneumonia_files = []

test_normal_files = []

test_pneumonia_files = []



for dirname, _, filenames in os.walk('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL'):

    for filename in filenames:

        train_normal_files.append(os.path.join(dirname, filename))

        

for dirname, _, filenames in os.walk('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/PNEUMONIA'):

    for filename in filenames:

        train_pneumonia_files.append(os.path.join(dirname, filename))        

        

for dirname, _, filenames in os.walk('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/NORMAL'):

    for filename in filenames:

        test_normal_files.append(os.path.join(dirname, filename))

        

for dirname, _, filenames in os.walk('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/PNEUMONIA'):

    for filename in filenames:

        test_pneumonia_files.append(os.path.join(dirname, filename))  



# Any results you write to the current directory are saved as output.
%matplotlib inline
# Let's explore our training dataset



# Get number of samples for each class (normal/pneumonia)

print("normal sample count:", len(train_normal_files))

print("pneumonia sample count:", len(train_pneumonia_files))



sample = imageio.imread(train_normal_files[0])

print("sample dimensions: ", sample.shape)

print("min and max values: ", np.min(sample), np.max(sample))

temp_shape = sample.shape



# View the first few samples

for indx in range(5):

    plt.figure() # don't overwrite other plots

    sample = imageio.imread(train_normal_files[indx])

    plt.imshow(sample)

    

# View the first few samples

for indx in range(5):

    plt.figure() # don't overwrite other plots

    sample = imageio.imread(train_pneumonia_files[indx])

    plt.imshow(sample)

    

# From this we have a rough idea what the images look like, and we know that the samples are nonunifrom sizes with nonuniform channels.
dims = (50, 50, 3)
from skimage.transform import resize



# train

train_normal = []

train_normal_dims = []

for indx in range(len(train_normal_files)):

    try:

        sample = imageio.imread(train_normal_files[indx])

        sample = np.array(sample)

        sample = resize(sample, dims) # reshape so all images are same dimensions

        train_normal_dims.append(sample)

        train_normal.append(sample.flatten()) # flatten each sample so it's the correct shape for umap clustering

    except:

        print("> error loading image: ", train_normal_files[indx])

print("normal train count:", len(train_normal))

    

train_pneumonia = []

train_pneumonia_dims = []

for indx in range(len(train_pneumonia_files)):

    try:

        sample = imageio.imread(train_pneumonia_files[indx])

        sample = np.array(sample)

        sample = resize(sample, dims) # reshape so all images are same dimensions

        train_pneumonia_dims.append(sample)

        train_pneumonia.append(sample.flatten()) # flatten each sample so it's the correct shape for umap clustering

    except:

        print("> error loading image: ", train_pneumonia_files[indx])

print("pneumonia train count:", len(train_pneumonia))



# test data



test_normal = []

test_normal_dims = []

for indx in range(len(test_normal_files)):

    try:

        sample = imageio.imread(test_normal_files[indx])

        sample = np.array(sample)

        sample = resize(sample, dims) # reshape so all images are same dimensions

        test_normal_dims.append(sample)

        test_normal.append(sample.flatten()) # flatten each sample so it's the correct shape for umap clustering

    except:

        print("> error loading image: ", test_normal_files[indx])

print("normal test count:", len(test_normal))

    

test_pneumonia = []

test_pneumonia_dims = []

for indx in range(len(test_pneumonia_files)):

    try:

        sample = imageio.imread(test_pneumonia_files[indx])

        sample = np.array(sample)

        sample = resize(sample, dims) # reshape so all images are same dimensions

        test_pneumonia_dims.append(sample)

        test_pneumonia.append(sample.flatten()) # flatten each sample so it's the correct shape for umap clustering

    except:

        print("> error loading image: ", test_pneumonia_files[indx])

print("pneumonia test count:", len(test_pneumonia))
# View the first few samples

for indx in range(5):

    plt.figure() # don't overwrite other plots

    plt.imshow((train_pneumonia_dims[indx]))
!pip install umap-learn==0.4
# Get the data in a more traditional ML processing format (data and label arrays)

n = None #if dataset were too large for processing, n can be used to specify subset size to use



# train

print(len(train_normal), len(train_pneumonia))



if n is not None:

    train_normal_labels = np.zeros(len(train_normal[:n]))

    train_pneumonia_labels = np.ones(len(train_pneumonia[:n]))

    target = np.concatenate([train_normal_labels[:n], train_pneumonia_labels[:n]])



    data = train_normal[:n] + train_pneumonia[:n]

    data_dims = train_normal_dims[:n] + train_pneumonia_dims[:n]

else:

    train_normal_labels = np.zeros(len(train_normal))

    train_pneumonia_labels = np.ones(len(train_pneumonia))

    target = np.concatenate([train_normal_labels, train_pneumonia_labels])



    data = train_normal + train_pneumonia

    data_dims = train_normal_dims + train_pneumonia_dims

data = np.array(data)

data_dims = np.array(data_dims)



from sklearn.utils import shuffle

data, data_dims, target = shuffle(data, data_dims, target, random_state=0)



# test

print(len(test_normal), len(test_pneumonia))



if n is not None:

    test_normal_labels = np.zeros(len(test_normal[:n]))

    test_pneumonia_labels = np.ones(len(test_pneumonia[:n]))

    test_target = np.concatenate([test_normal_labels[:n], test_pneumonia_labels[:n]])



    test_data = test_normal[:n] + test_pneumonia[:n]

    test_data_dims = test_normal_dims[:n] + test_pneumonia_dims[:n]

else:

    test_normal_labels = np.zeros(len(test_normal))

    test_pneumonia_labels = np.ones(len(test_pneumonia))

    test_target = np.concatenate([test_normal_labels, test_pneumonia_labels])



    test_data = test_normal + test_pneumonia

    test_data_dims = test_normal_dims + test_pneumonia_dims

test_data = np.array(test_data)

test_data_dims = np.array(test_data_dims)



from sklearn.utils import shuffle

test_data, test_data_dims, test_target = shuffle(test_data, test_data_dims, test_target, random_state=0)





print("train:", data.shape, data_dims.shape, target.shape)

print("test:", test_data.shape, test_data_dims.shape, test_target.shape)
# Let's try some dimensionality reduction based clustering: UMAP



import umap

print(umap.__version__)



reducer = umap.UMAP(low_memory=True)

embedding = reducer.fit_transform(data)



fig, ax = plt.subplots(figsize=(12, 10))

color = target

plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=2)

plt.setp(ax, xticks=[], yticks=[])

plt.title("Pneumonia data embedded into two dimensions by UMAP", fontsize=18)



plt.show()
pneumonia = pd.DataFrame({"x":embedding[:, 0], "y":embedding[:, 1], "color":target})#, "image":data})
import plotly.express as px



fig = px.scatter(pneumonia, x="x", y="y", color="color", title="Pneumonia data embedded into two dimensions by UMAP")



fig.show()
# embed the test data



test_embedding = reducer.transform(test_data)



#total_embedding = embedding + test_embedding



fig, ax = plt.subplots(figsize=(12, 10))

color = test_target

plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=color, cmap="Spectral", s=2)

plt.setp(ax, xticks=[], yticks=[])

plt.title("Test Pneumonia data embedded into two dimensions by UMAP", fontsize=18)



plt.show()
# build some classifiers on the embeddings and check their performance



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



svc = SVC().fit(embedding, target)

knn = KNeighborsClassifier().fit(embedding, target)



print("Support Vector score:", svc.score(test_embedding, test_target))

print("K-Nearest Neighbor score:", knn.score(test_embedding, test_target))
# keep more dimensions (5) during dimensionality reduction and compare performance



reducer_5 = umap.UMAP(low_memory=True, n_components=5)

embedding_5 = reducer_5.fit_transform(data)

test_embedding_5 = reducer_5.transform(test_data)



svc_5 = SVC().fit(embedding_5, target)

knn_5 = KNeighborsClassifier().fit(embedding_5, target)



print("Support Vector (5 component) score:", svc_5.score(test_embedding_5, test_target))

print("K-Nearest Neighbor (5 component) score:", knn_5.score(test_embedding_5, test_target))
# keep more dimensions (10) during dimensionality reduction and compare performance



reducer_10 = umap.UMAP(low_memory=True, n_components=10)

embedding_10 = reducer_10.fit_transform(data)

test_embedding_10 = reducer_10.transform(test_data)



svc_10 = SVC().fit(embedding_10, target)

knn_10 = KNeighborsClassifier().fit(embedding_10, target)



print("Support Vector (10 component) score:", svc_10.score(test_embedding_10, test_target))

print("K-Nearest Neighbor (10 component) score:", knn_10.score(test_embedding_10, test_target))