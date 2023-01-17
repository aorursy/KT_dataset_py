import random

import numpy as np

import matplotlib.pyplot as plt





%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# for auto-reloading extenrnal modules

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2
# Get CIFAR10 data

!pip install wget

import wget

import tarfile

import os



filename = "cifar-10-batches-py/data_batch_1"

if not os.path.exists(filename):

    path = "data.tar.gz"

    if not os.path.exists(path):

        wget.download("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", path)

    tar = tarfile.open(path, "r:gz")

    tar.extractall()

    tar.close()

# Loading data in X and Y numpy array

# dict_classes maps Y values to its class name

import pickle



""" load single batch of cifar """

dict_classes = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

X = None

y = None

with open(filename, 'rb') as f:

    datadict = pickle.load(f, encoding='latin1')

    X = datadict['data']

    y = datadict['labels']

    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")

    y = np.array(y)

    

""" X is numpy array images (samples_size, X_Dim, Y_Dim, Channels), y contains its corresponding labels """
# check the X shape

print(X.shape)
# importing the collections module

import collections

# getting the elements frequencies using Counter class

elements_count = collections.Counter(y)

# printing the element and the frequency

for key, value in elements_count.items():

    print(f"{key}: {value}")
def visualize_images(examples_per_class):

    """Joins all 3 data's dataframe."""

    #INPUT: examples_per_class: number of examples of each class you want to show

    nbr_each_class = examples_per_class

    img_class = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

    nbr_class = len(img_class)

    

    for i,cls in enumerate(img_class):

        index = np.flatnonzero(y == i)

        index = np.random.choice(index,nbr_each_class,replace=False)

        for j,index in enumerate(index):

            plt_index = j * nbr_class+(i+1)

            plt.subplot(nbr_each_class,nbr_class,plt_index)

            plt.imshow(X[index].astype('uint8'))

            plt.axis('off')

            if j == 0:

                plt.title(cls)
visualize_images(5)
# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)

try:

   del X_train, y_train

   del X_test, y_test

   del X_val, y_val

   print('Clear previously loaded data.')

except:

   pass



from sklearn.model_selection import train_test_split

def get_train_val_test(num_training, num_validation, num_test):

    # Split data and return train, test and validation splits

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1000, random_state = 0)

    

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1000, random_state=1)



    return X_train, y_train, X_val, y_val, X_test, y_test



X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(8000, 1000, 1000)

print("Training dataset size", len(X_train))

print("Validation dataset size", len(X_val))

print("Test dataset size", len(X_test))
# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)

try:

   del X_train_feats

   del X_val_feats

   del X_test_feats

   print('Clear previously loaded data.')

except:

   pass

from features import *

num_color_bins = 10 # Number of bins in the color histogram

feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]

X_train_feats = extract_features(X_train, feature_fns, verbose=True)

X_val_feats = extract_features(X_val, feature_fns)

X_test_feats = extract_features(X_test, feature_fns)

print("Shape of X_train_feats features", X_train_feats.shape)

print("Shape of X_val_feats features", X_val_feats.shape)

print("Shape of X_test_feats features", X_test_feats.shape)
# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)

try:

   del X_train_feat_norm

   del X_val_feat_norm

   del X_feat_test_norm

   print('Clear previously loaded data.')

except:

   pass



from sklearn.preprocessing import StandardScaler

def get_normalized_features(X_train, X_val, X_test):

    # returns the normalized X_train, X_val and X_test

    # INPUT: X_train : Training data features

    #        X_val : Validation data features

    #        X_test : Test data features

    # RETURN: X_train_norm, X_val_norm, X_test_norm

    

    # TODO: Your code here   

    scaler = StandardScaler()

    X_train_norm = scaler.fit_transform(X_train)

    X_val_norm = scaler.transform(X_val)

    X_test_norm = scaler.transform(X_test)

    # End of your code



    return X_train_norm, X_val_norm, X_test_norm



X_train_feat_norm, X_val_feat_norm, X_feat_test_norm = get_normalized_features(X_train_feats, X_val_feats, X_test_feats)

print("Shape of X_train_feat_norm features", X_train_feat_norm.shape)
%%time

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import GridSearchCV



np.random.seed(0)

best_knn = None

knn_best_val = -1



################################################################################

# TODO:                                                                        #

# Use the validation set to find the best fit parameters of the KNN algorithm. #

# Save the best validation score in  knn_best_val.                             #

# Save the best trained classifer in best_knn.                                 #

################################################################################

param_grid = [{"n_neighbors":[i for i in range(1,15)]}]

knn_clf = KNeighborsClassifier()

grid_search = GridSearchCV(knn_clf,param_grid,cv = 5,n_jobs = -1, verbose = 2,scoring = 'accuracy')

grid_search.fit(X_val_feat_norm,y_val)

best_knn = grid_search.best_estimator_

knn_best_val = grid_search.best_score_

################################################################################

#                              END OF YOUR CODE                                #

################################################################################ 



print('Best validation accuracy achieved during cross-validation: %f' % knn_best_val)

print("Paramters of best trained KNN", best_knn.get_params())
# Define and train the KNN with best fitted parameters



# TODO: Your code here   

knn = KNeighborsClassifier(n_neighbors=6)

knn_model_1 = knn.fit(X_train_feat_norm, y_train)



y_true, y_pred = y_test, knn_model_1.predict(X_feat_test_norm)

test_acc_knn = knn_model_1.score(X_feat_test_norm, y_test)

# End of your code



print("Test accuracy = ", test_acc_knn * 100)

# Print Accuracy and Classification Report on test data



# TODO: Your code here   

from sklearn.metrics import classification_report

y_true, y_pred = y_test, knn_model_1.predict(X_feat_test_norm)

class_report_knn=classification_report(y_true, y_pred)

# End of your code



print(class_report_knn)

from sklearn import svm, grid_search

from sklearn.svm import SVC

np.random.seed(0)

svm_best_val = -1

best_svm = None



################################################################################

# TODO:                                                                        #

# Use the validation set to find the best fit parameters of the SVM algorithm. #

# Save the best validation score in  svm_best_val.                             #

# Save the best trained classifer in best_svm.                                 #

################################################################################



Cs = [0.001, 0.01, 0.1, 1, 10]

gammas = [0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)

grid_search.fit(X_val_feat_norm,y_val)

best_svm = grid_search.best_params_

svm_best_val = grid_search.best_score_



################################################################################

#                              END OF YOUR CODE                                #

################################################################################



print('best validation accuracy achieved during cross-validation: %f' % svm_best_val)

print("Paramters of best trained SVM",grid_search.best_params_)
# TODO: Your code here   

svm = svm.SVC(C=10, kernel='rbf',gamma=0.01)

svm_model = svm.fit(X_train_feat_norm, y_train)



y_true, y_pred = y_test, svm_model.predict(X_feat_test_norm)

test_acc_svm = svm_model.score(X_feat_test_norm, y_test)

# End of your code



print("Test accuracy = ", test_acc_svm * 100)

# TODO: Your code here   

y_true, y_pred = y_test, svm_model.predict(X_feat_test_norm)

class_report_svm=classification_report(y_true, y_pred)

# End of your code



print(class_report_svm)

def visualize_negative_classified_images(examples_per_class):

    """Joins all 3 data's dataframe."""

    #INPUT: examples_per_class: number of examples of each class you want to show

    nbr_each_class = examples_per_class

    img_class = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

    nbr_class = len(img_class)

    

    for i,cls in enumerate(img_class):

        index = np.flatnonzero(y_test[y_pred != y_test]==i)

        index = np.random.choice(index,nbr_each_class,replace=False)

        for j,index in enumerate(index):

            plt_index = j * nbr_class+(i+1)

            plt.subplot(nbr_each_class,nbr_class,plt_index)

            plt.imshow(X_test[index].astype('uint8'))

            plt.axis('off')

            if j == 0:

                plt.title(cls)



visualize_negative_classified_images(5)