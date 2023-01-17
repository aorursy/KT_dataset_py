# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import cv2

import matplotlib.pyplot as plt

import os

import seaborn as sns

import umap

from PIL import Image

from scipy import misc

from os import listdir

from os.path import isfile, join

import numpy as np

from scipy import misc

from random import shuffle

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.utils.np_utils import to_categorical

os.chdir('../input/utkface_aligned_cropped')
#os.listdir()

#os.chdir('input')

os.listdir()
os.chdir('crop_part1')

os.listdir()[:5]
im =Image.open('27_0_1_20170102233552626.jpg.chip.jpg').resize((128,128))

im
onlyfiles = os.listdir()
len(onlyfiles)



#Enable Asian only

asian = []

for name in onlyfiles:

    race = name.split('_')[2]

    if race == '2':

        asian.append(name)

onlyfiles = asian
shuffle(onlyfiles)

age = [i.split('_')[0] for i in onlyfiles]
class_label = ['17-','18-24','25-34','35-44','45-60','60+']





# classes = []

# for i in age:

#     i = int(i)

#     if i <= 17:

#         classes.append(0)

#     if (i>17) and (i<=24):

#         classes.append(1)

#     if (i>25) and (i<34):

#         classes.append(2)

#     if (i>=45) and (i<60):

#         classes.append(3)

#     if i>=60:

#         classes.append(4)





classes = []

Y_age = []

for i in age:

    i = int(i)

    if i <= 17:

        classes.append(0)

    elif (i>17) and (i<=24):

        classes.append(1)

    elif (i>24) and (i<=34):

        classes.append(2)

    elif (i>34) and (i<=44):

        classes.append(3)

    elif (i>44) and (i<=60):

        classes.append(4)

    elif i>60:

        classes.append(5)

    Y_age.append(i)
# X_data =[]

# for file in onlyfiles:

#     face = misc.imread(file)

#     face =cv2.resize(face, (32, 32) )

#     X_data.append(face)



# This way is faster than the original one

def convertImage(filename):

    face = misc.imread(filename)

    face = cv2.resize(face, (32, 32))

    return face 

X_data = list(map(convertImage, onlyfiles))

X = np.squeeze(X_data)

X.shape
# normalize data

X = X.astype('float32')

X /= 255

classes[:10]

categorical_labels = to_categorical(classes, num_classes=6)

categorical_labels
categorical_labels[:10]

len(X)
(x_train, y_train, y_train_age), (x_test, y_test, y_test_age) = (X[:1100],categorical_labels[:1100], Y_age[:1100]) , (X[1100:] , categorical_labels[1100:], Y_age[1100:])

#(x_valid , y_valid) = (x_test[1000:], y_test[1000:])

#(x_test, y_test) = (x_test[:1000], y_test[:1000])

len(x_train)+len(x_test)  == len(X)

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, classification_report


cv2.cvtColor(x_train[0], cv2.COLOR_BGR2GRAY).shape
x_train_temp = np.array([cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY).flatten() for x_t in x_train]) 

x_train_temp.shape
x_train_df = pd.DataFrame(x_train_temp)

x_train_df
def plot_faces(pixels):

    fig, axes = plt.subplots(5, 5, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):

        ax.imshow(np.array(pixels)[i].reshape(32, 32), cmap='gray')

    plt.show()

# for x_t in x_train_temp:

#     plot_faces(x_t)

#     break

plot_faces(x_train_df)
pca = PCA().fit(x_train_df)

plt.figure(figsize=(18, 7))

plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3)
representPercesntage = 0.999 # This significantly affects accuracy

#representPercesntage = 0.8

np.where(pca.explained_variance_ratio_.cumsum() > representPercesntage)[0][0:10]
n_com = np.where(pca.explained_variance_ratio_.cumsum() > representPercesntage)[0][0]
pca = PCA(n_components=n_com).fit(x_train_df)
x_train_pca = pca.transform(x_train_df)
x_train_pca.shape
y_train_sklearn = np.array([np.where(yt == 1)[0][0] for yt in y_train])
y_train_sklearn
classifier = SVC().fit(x_train_pca,y_train_sklearn)
predictions = classifier.predict(x_train_pca)

target_names = [str(l) for l in range(6)]

print(classification_report(y_train_sklearn, predictions, target_names=target_names))
TruePrediction = 0

for i in range(len(predictions)):

    if predictions[i] == y_train_sklearn[i]:

        TruePrediction += 1

print(f"Accuracy: {TruePrediction/len(predictions)}")
# Preprocessing pipeline

x_test_temp = np.array([cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY).flatten() for x_t in x_test]) 

x_test_df = pd.DataFrame(x_test_temp)

x_test_pca = pca.transform(x_test_df)



predictions = classifier.predict(x_test_pca)



y_test_sklearn = np.array([np.where(yt == 1)[0][0] for yt in y_test])



print(classification_report(y_test_sklearn, predictions))
TruePrediction = 0

loss = 0

N = len(predictions)

for i in range(len(predictions)):

    if predictions[i] == y_test_sklearn[i]:

        TruePrediction += 1

    else:

        loss += (y_test_sklearn[i] - predictions[i])**2

print(f"Accuracy: {TruePrediction/len(predictions)}")

print(f"Loss: {(loss/N)**0.5}")
labels = class_label





# Plot a random sample of 10 test images, their predicted labels and ground truth

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(x_test[index]))

    predict_index = predictions[index]

    true_index = np.argmax(y_test[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(labels[predict_index], 

                                  labels[true_index]),

                                  color=("green" if predict_index == true_index else "red"))

plt.show()
#SVC linear



classifier = SVC(kernel = 'linear').fit(x_train_pca,y_train_sklearn)

# Preprocessing pipeline

x_test_temp = np.array([cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY).flatten() for x_t in x_test]) 

x_test_df = pd.DataFrame(x_test_temp)

x_test_pca = pca.transform(x_test_df)



predictions = classifier.predict(x_test_pca)



y_test_sklearn = np.array([np.where(yt == 1)[0][0] for yt in y_test])



print(classification_report(y_test_sklearn, predictions))



TruePrediction = 0

loss = 0

N = len(predictions)

for i in range(N):

    if predictions[i] == y_test_sklearn[i]:

        TruePrediction += 1

    else:

        loss += (y_test_sklearn[i] - predictions[i])**2

print(f"Accuracy: {TruePrediction/len(predictions)}")

print(f"Loss: {(loss/N)**0.5}")



# Plot a random sample of 10 test images, their predicted labels and ground truth

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(x_test[index]))

    predict_index = predictions[index]

    true_index = np.argmax(y_test[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(labels[predict_index], 

                                  labels[true_index]),

                                  color=("green" if predict_index == true_index else "red"))

plt.show()
#SVC poly



classifier = SVC(kernel = 'poly', gamma='auto', class_weight='balanced').fit(x_train_pca,y_train_sklearn)

# Preprocessing pipeline

x_test_temp = np.array([cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY).flatten() for x_t in x_test]) 

x_test_df = pd.DataFrame(x_test_temp)

x_test_pca = pca.transform(x_test_df)



predictions = classifier.predict(x_test_pca)



y_test_sklearn = np.array([np.where(yt == 1)[0][0] for yt in y_test])



print(classification_report(y_test_sklearn, predictions))



TruePrediction = 0

loss = 0

N = len(predictions)

for i in range(N):

    if predictions[i] == y_test_sklearn[i]:

        TruePrediction += 1

    else:

        loss += (y_test_sklearn[i] - predictions[i])**2

print(f"Accuracy: {TruePrediction/len(predictions)}")

print(f"Loss: {(loss/N)**0.5}")





# Plot a random sample of 10 test images, their predicted labels and ground truth

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(x_test[index]))

    predict_index = predictions[index]

    true_index = np.argmax(y_test[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(labels[predict_index], 

                                  labels[true_index]),

                                  color=("green" if predict_index == true_index else "red"))

plt.show()
from sklearn.cluster import KMeans

import numpy as np



kmeans = KMeans(n_clusters=6, random_state=2).fit(x_train_pca)

kmeans.labels_

results = {

    i:[] for i in range(6)

}



colors = ['r','g','b','y','m','c']

for i in range(len(x_train_pca)):

    pc1, pc2 = x_train_pca[i][0:2]

    label = kmeans.labels_[i]

    results[label].append(y_train_sklearn[i])

    plt.scatter(pc1, pc2, c= colors[label],

            s=50, cmap='viridis');

    
from collections import Counter

sumCounter = Counter()

for k,v in results.items():

    print(f"Label {k}")

    print(Counter(v))

    sumCounter += Counter(v)
sumCounter
from sklearn.linear_model import LinearRegression



reg = LinearRegression().fit(x_train_pca, y_train_age)

reg.score(x_train_pca, y_train_age)



predictions = reg.predict(x_train_pca)



accuracy = 0

for i in range(len(predictions)):

    print(f"Prediction: {predictions[i]}")

    print(f"Real: {y_train_age[i]}")

    

    if predictions[i] <= 17:

        Class = 0

    elif (predictions[i]>17) and (predictions[i]<=24):

        Class = 1

    elif (predictions[i]>24) and (predictions[i]<=34):

        Class = 2

    elif (predictions[i]>34) and (predictions[i]<=44):

        Class = 3

    elif (predictions[i]>44) and (predictions[i]<=60):

        Class = 4

    elif predictions[i]>60:

        Class = 5

    if Class == y_train_sklearn[i]:

        accuracy += 1

accuracy /= len(predictions)

print(f"Train accuracy {accuracy}")



predictions = reg.predict(x_test_pca)

predictions_class = []

accuracy = 0

for i in range(len(predictions)):

    print(f"Prediction: {predictions[i]}")

    print(f"Real: {y_test_age[i]}")

    

    if predictions[i] <= 17:

        Class = 0

    elif (predictions[i]>17) and (predictions[i]<=24):

        Class = 1

    elif (predictions[i]>24) and (predictions[i]<=34):

        Class = 2

    elif (predictions[i]>34) and (predictions[i]<=44):

        Class = 3

    elif (predictions[i]>44) and (predictions[i]<=60):

        Class = 4

    elif predictions[i]>60:

        Class = 5

    if Class == y_test_sklearn[i]:

        accuracy += 1

    predictions_class.append(Class)

accuracy /= len(predictions)

print(f"Test accuracy {accuracy}")
# Plot a random sample of 10 test images, their predicted labels and ground truth

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(x_test[index]))

    predict_index = predictions_class[index]

    true_index = np.argmax(y_test[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(labels[predict_index], 

                                  labels[true_index]),

                                  color=("green" if predict_index == true_index else "red"))

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
clf = RandomForestClassifier(max_depth=100, random_state=100)

clf.fit(x_train_pca, y_train_sklearn)
predictions = clf.predict(x_train_pca)

print(classification_report(y_train_sklearn, predictions))



TruePrediction = 0

loss = 0

N = len(predictions)

for i in range(N):

    if predictions[i] == y_train_sklearn[i]:

        TruePrediction += 1

    else:

        loss += (y_train_sklearn[i] - predictions[i])**2

print(f"Accuracy: {TruePrediction/len(predictions)}")

print(f"Loss: {(loss/N)**0.5}")



predictions = clf.predict(x_test_pca)

print(classification_report(y_test_sklearn, predictions))





TruePrediction = 0

loss = 0

N = len(predictions)

for i in range(N):

    if predictions[i] == y_test_sklearn[i]:

        TruePrediction += 1

    else:

        loss += (y_test_sklearn[i] - predictions[i])**2

print(f"Accuracy: {TruePrediction/len(predictions)}")

print(f"Loss: {(loss/N)**0.5}")



# Plot a random sample of 10 test images, their predicted labels and ground truth

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(x_test[index]))

    predict_index = predictions[index]

    true_index = np.argmax(y_test[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(labels[predict_index], 

                                  labels[true_index]),

                                  color=("green" if predict_index == true_index else "red"))

plt.show()
# Plot a random sample of 10 test images, their predicted labels and ground truth

figure = plt.figure(figsize=(20, 8))

for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(x_test[index]))

    predict_index = predictions[index]

    true_index = np.argmax(y_test[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(labels[predict_index], 

                                  labels[true_index]),

                                  color=("green" if predict_index == true_index else "red"))

plt.show()