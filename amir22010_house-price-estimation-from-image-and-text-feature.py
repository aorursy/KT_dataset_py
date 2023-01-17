# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/repository/emanhamed-Houses-dataset-41e7392/"))



# Any results you write to the current directory are saved as output.
# import the necessary packages

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import numpy as np

import glob

import cv2

import os



def load_house_attributes(inputPath):

	# initialize the list of column names in the CSV file and then

	# load it using Pandas

	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]

	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)



	# determine (1) the unique zip codes and (2) the number of data

	# points with each zip code

	zipcodes = df["zipcode"].value_counts().keys().tolist()

	counts = df["zipcode"].value_counts().tolist()



	# loop over each of the unique zip codes and their corresponding

	# count

	for (zipcode, count) in zip(zipcodes, counts):

		# the zip code counts for our housing dataset is *extremely*

		# unbalanced (some only having 1 or 2 houses per zip code)

		# so let's sanitize our data by removing any houses with less

		# than 25 houses per zip code

		if count < 25:

			idxs = df[df["zipcode"] == zipcode].index

			df.drop(idxs, inplace=True)



	# return the data frame

	return df



def process_house_attributes(df, train, test):

	# initialize the column names of the continuous data

	continuous = ["bedrooms", "bathrooms", "area"]



	# performin min-max scaling each continuous feature column to

	# the range [0, 1]

	cs = MinMaxScaler()

	trainContinuous = cs.fit_transform(train[continuous])

	testContinuous = cs.transform(test[continuous])



	# one-hot encode the zip code categorical data (by definition of

	# one-hot encoing, all output features are now in the range [0, 1])

	zipBinarizer = LabelBinarizer().fit(df["zipcode"])

	trainCategorical = zipBinarizer.transform(train["zipcode"])

	testCategorical = zipBinarizer.transform(test["zipcode"])



	# construct our training and testing data points by concatenating

	# the categorical features with the continuous features

	trainX = np.hstack([trainCategorical, trainContinuous])

	testX = np.hstack([testCategorical, testContinuous])



	# return the concatenated training and testing data

	return (trainX, testX)



def load_house_images(df, inputPath):

	# initialize our images array (i.e., the house images themselves)

	images = []



	# loop over the indexes of the houses

	for i in df.index.values:

		# find the four images for the house and sort the file paths,

		# ensuring the four are always in the *same order*

		basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])

		housePaths = sorted(list(glob.glob(basePath)))



		# initialize our list of input images along with the output image

		# after *combining* the four input images

		inputImages = []

		outputImage = np.zeros((64, 64, 3), dtype="uint8")



		# loop over the input house paths

		for housePath in housePaths:

			# load the input image, resize it to be 32 32, and then

			# update the list of input images

			image = cv2.imread(housePath)

			image = cv2.resize(image, (32, 32))

			inputImages.append(image)



		# tile the four input images in the output image such the first

		# image goes in the top-right corner, the second image in the

		# top-left corner, the third image in the bottom-right corner,

		# and the final image in the bottom-left corner

		outputImage[0:32, 0:32] = inputImages[0]

		outputImage[0:32, 32:64] = inputImages[1]

		outputImage[32:64, 32:64] = inputImages[2]

		outputImage[32:64, 0:32] = inputImages[3]



		# add the tiled image to our set of images the network will be

		# trained on

		images.append(outputImage)



	# return our set of images

	return np.array(images)
# import the necessary packages

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import numpy as np

import os



# construct the path to the input .txt file that contains information

# on each house in the dataset and then load the dataset

print("[INFO] loading house attributes...")

inputPath = os.path.sep.join(["../input/repository/emanhamed-Houses-dataset-41e7392/Houses Dataset/","HousesInfo.txt"]) 

df = load_house_attributes(inputPath)
# load the house images and then scale the pixel intensities to the

# range [0, 1]

print("[INFO] loading house images...")

images = load_house_images(df, "../input/repository/emanhamed-Houses-dataset-41e7392/Houses Dataset/")

images = images / 255.0
# partition the data into training and testing splits using 75% of

# the data for training and the remaining 25% for testing

split = train_test_split(df, images, test_size=0.25, random_state=42)

(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
# find the largest house price in the training set and use it to

# scale our house prices to the range [0, 1] (will lead to better

# training and convergence)

maxPrice = trainAttrX["price"].max()

trainY = trainAttrX["price"] / maxPrice

testY = testAttrX["price"] / maxPrice
# import the necessary packages

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.layers import Flatten

from keras.layers import Input

from keras.models import Model



def create_mlp(dim, regress=False):

	# define our MLP network

	model = Sequential()

	model.add(Dense(8, input_dim=dim, activation="relu"))

	model.add(Dense(4, activation="relu"))



	# check to see if the regression node should be added

	if regress:

		model.add(Dense(1, activation="linear"))



	# return our model

	return model



def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):

	# initialize the input shape and channel dimension, assuming

	# TensorFlow/channels-last ordering

	inputShape = (height, width, depth)

	chanDim = -1



	# define the model input

	inputs = Input(shape=inputShape)



	# loop over the number of filters

	for (i, f) in enumerate(filters):

		# if this is the first CONV layer then set the input

		# appropriately

		if i == 0:

			x = inputs



		# CONV => RELU => BN => POOL

		x = Conv2D(f, (3, 3), padding="same")(x)

		x = Activation("relu")(x)

		x = BatchNormalization(axis=chanDim)(x)

		x = MaxPooling2D(pool_size=(2, 2))(x)



	# flatten the volume, then FC => RELU => BN => DROPOUT

	x = Flatten()(x)

	x = Dense(16)(x)

	x = Activation("relu")(x)

	x = BatchNormalization(axis=chanDim)(x)

	x = Dropout(0.5)(x)



	# apply another FC layer, this one to match the number of nodes

	# coming out of the MLP

	x = Dense(4)(x)

	x = Activation("relu")(x)



	# check to see if the regression node should be added

	if regress:

		x = Dense(1, activation="linear")(x)



	# construct the CNN

	model = Model(inputs, x)



	# return the CNN

	return model
# create our Convolutional Neural Network and then compile the model

# using mean absolute percentage error as our loss, implying that we

# seek to minimize the absolute percentage difference between our

# price *predictions* and the *actual prices*

model = create_cnn(64, 64, 3, regress=True)

opt = Adam(lr=1e-3, decay=1e-3 / 200)

model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
# train the model

print("[INFO] training model...")

model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),

	epochs=200, batch_size=8)
# make predictions on the testing data

print("[INFO] predicting house prices...")

preds = model.predict(testImagesX)



# compute the difference between the *predicted* house prices and the

# *actual* house prices, then compute the percentage difference and

# the absolute percentage difference

diff = preds.flatten() - testY

percentDiff = (diff / testY) * 100

absPercentDiff = np.abs(percentDiff)



# compute the mean and standard deviation of the absolute percentage

# difference

mean = np.mean(absPercentDiff)

std = np.std(absPercentDiff)



# finally, show some statistics on our model

from babel.numbers import format_currency

print("[INFO] avg. house price: {}, std house price: {}".format(

	format_currency(df["price"].mean(), 'USD', locale='en_US'),

	format_currency(df["price"].std(), 'USD', locale='en_US')))