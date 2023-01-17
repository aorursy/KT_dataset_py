# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# mathematical dependencies
import numpy as np

# plotting dependencies
from matplotlib import pyplot as plt

# image processing dependencies
import cv2
from keras.preprocessing.image import img_to_array, load_img
from sklearn.datasets import load_files

# Utilities / tools
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# GridSearch for fitting & tunning
from sklearn.model_selection import GridSearchCV

# Hierarchical Data Format is a set of file formats designed to store and organize large amounts of data.
# Used in order to export the formatted dataset to google colab, kaggle or any other cloud solution
import h5py
# ML Methods and tools

# Multi-layer Perceptron classifier.
from sklearn.neural_network import MLPClassifier
# k-NN classifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Compare random dummy classifiers with our main methods
from sklearn.dummy import DummyClassifier
train_directory = '/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training/'
test_directory = '/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test/'
# Display a colored image with the given size  (X and Y axis)
# The default values are x=50 and y=50
def display_image(image, x=50, y=50):
    #image = np.array(image, dtype='float')
    pixels = image.reshape((x, y))
    #plt.imshow(pixels, cmap='gray')
    plt.imshow(pixels)
    plt.show()
# MinMaxScaler will transform each value in the column proportionally within the 0,1 range 
# The shape of the initial dataset will be preserved
def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)
# Function returning two arrays:
# An array with the path to each image.
# An array with the label for each image.
def load_data_information(path):
    # values are shuffled by default
    data = load_files(path)

    # Array of file names
    files = np.array(data['filenames'])
   
    # Array containing the labels associated with the image at the same position in the files array
    file_labels = np.array(data['target'])
   
    # Array of labels - the name of each label / unused for the moment, but good to have.
    labels = np.array(data['target_names'])

    return files, file_labels
# Given an array which contains the file path of each image,
# this function will return a numpy array with each image loaded into the memory
def obtain_files_from_path(files, flatten = False):
    images_as_array=[]
    for file in files:
        # ignore OS specific files
        if not file.startswith('.') and file != 'Thumbs.db':
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            if flatten:
                image = image.flatten()
            images_as_array.append(image)
    return np.array(images_as_array)
training_files_path, training_labels = load_data_information(train_directory)
test_files_path, test_labels = load_data_information(test_directory)
training_images = obtain_files_from_path(training_files_path, True)
test_images = obtain_files_from_path(test_files_path, True)
training_images = scale_data(training_images)
test_images = scale_data(test_images)
print(training_images.shape)
print(test_images.shape)
def reduce_dimensionality2(data, nfeatures, reducer):
        algorithm = ""
        if reducer == 'PCA':
            algorithm = PCA(n_components=nfeatures)
        algorithm.fit(data)
        data = algorithm.transform(data)
        print("new total number of features {}".format(nfeatures))
        print("explained_variance_ratio_ {}".format(algorithm.explained_variance_ratio_.sum()))
        return data
training_imgs_pca = reduce_dimensionality2(training_images, 100, 'PCA')
test_imgs_pca     = reduce_dimensionality2(test_images, 100, 'PCA')
result = {}

for k in range(10, 20):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_images, training_labels)
    score = classifier.score(test_images, test_labels)
    result[k] = score
print(len(result))
print(list(result.values()))

x_min = 10
x = [i for i in range(x_min, x_min + len(result))]

plt.plot(x, list(result.values()), color = 'green')
plt.ylabel('Score')
plt.xlabel('Neighbours')
plt.title('KNN Performance for K with values between 10 and 20')
plt.show()
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(training_images, training_labels)
y_pred = knn.predict(test_images)
print(accuracy_score(test_labels, y_pred))
print(confusion_matrix(test_labels, y_pred))
print(training_images.shape)
print(training_images.shape)
mlp = MLPClassifier(hidden_layer_sizes=(512, 64, 128, 128, 1024), 
                    max_iter=300, 
                    learning_rate='adaptive',
                    solver='sgd', 
                    alpha=0.0002,  
                    activation='relu')
mlp.fit(training_images, training_labels)
predictions = mlp.predict(test_images)
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
dmc = DummyClassifier(strategy="most_frequent")
dmc.fit(training_images, training_labels)
result = dmc.score(test_images, test_labels)
print(result)
hyper_param_space_1 = {
    'hidden_layer_sizes': [(814,30,5), (16, 8, 8, 16), (64, 32, 32, 64)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.02],
    'learning_rate': ['constant','adaptive'],
}

hyper_param_space_2 = {
    'hidden_layer_sizes': [
(32, 64, 128, 512),
(64, 32, 128, 512),
(128, 32, 64, 512),
(32, 128, 64, 512),
(64, 128, 32, 512),
(128, 64, 32, 512),
(512, 64, 32, 128),
(64, 512, 32, 128),
(32, 512, 64, 128),
(512, 32, 64, 128),
(64, 32, 512, 128),
(32, 64, 512, 128),
(32, 128, 512, 64),
(128, 32, 512, 64),
(512, 32, 128, 64),
(32, 512, 128, 64),
(128, 512, 32, 64),
(512, 128, 32, 64),
(512, 128, 64, 32),
(128, 512, 64, 32),
(64, 512, 128, 32),
(512, 64, 128, 32),
(128, 64, 512, 32),
(64, 128, 512, 32)],
    'activation': ['relu'],
    'solver': ['sgd'],
    'alpha': [0.0002, 0.0001, 0.0004],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [0.001, 0.002, 0.003, 0.005, 0.009, 0.01]}
grid_search = GridSearchCV(mlp, hyper_param_space_2, n_jobs=-1, cv=3)
grid_search.fit(training_images, training_labels)
# Best paramete set
print('Best parameters found:', clf.best_params_)