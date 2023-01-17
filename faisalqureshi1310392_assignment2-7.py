#%matplotlib inline
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import cross_validation as cv
import skimage.feature as skimage
from skimage.filters import threshold_otsu
from skimage import img_as_bool
from skimage.morphology import skeletonize

import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Cross Validation function - takes data and model, returns average accuracy score for that model 
def cross_validate(data,target,model):
	num_folds = 10;
	kf = cv.KFold(len(data),num_folds)
	for train, test in kf:
		trainingDF = data.iloc[train].copy()
		testingDF = data.iloc[test].copy()
		model.fit(data,target)
		X = data
		t = target
		pred = model.predict(X)
		sum += metrics.accuracy_score(t,pred)
	return sum/num_folds

# ---------------- FEATURE EXTRACTION METHODS -----------------------
def hog_fe(df,num): # df is a [48k,1] shape containing 28x28 matrices
	hog_features_list = []
	for i in range(0,num):
		image = df[i][0]
		fd = hog(image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		hog_features_list.append(fd)
	hog_features = np.array(hog_features_list, 'float64')
	return hog_features
    

def corners_fe(df,num):
    corners = []
    for i in range(0,42000):
        image = df[i][0]
        corners_image = skimage.corner_fast(image, n=12, threshold=0.15)
        corners.append(corners_image)
    corners1 = np.array(corners, 'float64')
    return corners1

def cnn_fe():
    net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
    # layer parameters:
    input_shape=(None,1,28,28),
    hidden_num_units=1000, # number of units in 'hidden' layer
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.0001,
    update_momentum=0.9,

    max_epochs=15,
    verbose=1,
    )
    return net1

# -------------------- MAIN ----------------------
dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
#test = pd.read_csv("../input/test.csv").values

# Reshape each tuple into a 28x28 matrix:
target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
#test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)

# Pre-processing:
for i in range(0,42000):
    thresh = threshold_otsu(train[i][0])
    train[i][0] = train[i][0] > thresh
    train[i][0] = skeletonize(train[i][0])

features = hog_fe(train,42000)
#test = corners_fe(test,28000)

# Which model to use (un-comment which one you want to use )
#model = RandomForestClassifier(n_estimators=100)
def CNN(n_epochs):
    NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )
    return net1

cnn = CNN(15).fit(train,target) # train the CNN model for 15 epochs



#%matplotlib inline
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import cross_validation as cv
import skimage.feature as skimage
from skimage.filters import threshold_otsu
from skimage import img_as_bool
from skimage.morphology import skeletonize

import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Cross Validation function - takes data and model, returns average accuracy score for that model 
def cross_validate(data,target,model):
	num_folds = 10;
	kf = cv.KFold(len(data),num_folds)
	for train, test in kf:
		trainingDF = data.iloc[train].copy()
		testingDF = data.iloc[test].copy()
		model.fit(data,target)
		X = data
		t = target
		pred = model.predict(X)
		sum += metrics.accuracy_score(t,pred)
	return sum/num_folds

# ---------------- FEATURE EXTRACTION METHODS -----------------------
def hog_fe(df,num): # df is a [48k,1] shape containing 28x28 matrices
	hog_features_list = []
	for i in range(0,num):
		image = df[i][0]
		fd = hog(image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		hog_features_list.append(fd)
	hog_features = np.array(hog_features_list, 'float64')
	return hog_features
    

def corners_fe(df,num):
    corners = []
    for i in range(0,42000):
        image = df[i][0]
        corners_image = skimage.corner_fast(image, n=12, threshold=0.15)
        corners.append(corners_image)
    corners1 = np.array(corners, 'float64')
    return corners1

def cnn_fe():
    net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
    # layer parameters:
    input_shape=(None,1,28,28),
    hidden_num_units=1000, # number of units in 'hidden' layer
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.0001,
    update_momentum=0.9,

    max_epochs=15,
    verbose=1,
    )
    return net1

# -------------------- MAIN ----------------------
dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
#test = pd.read_csv("../input/test.csv").values

# Reshape each tuple into a 28x28 matrix:
target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
#test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)

# Pre-processing:
for i in range(0,42000):
    thresh = threshold_otsu(train[i][0])
    train[i][0] = train[i][0] > thresh
    train[i][0] = skeletonize(train[i][0])

features = hog_fe(train,42000)
#test = corners_fe(test,28000)

# Which model to use (un-comment which one you want to use )
#model = RandomForestClassifier(n_estimators=100)
def CNN(n_epochs):
    NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )
    return net1

cnn = CNN(15).fit(train,target) # train the CNN model for 15 epochs
