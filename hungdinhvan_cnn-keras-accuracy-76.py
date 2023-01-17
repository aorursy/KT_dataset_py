import tensorflow as tf 

import numpy as np 

from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPool2D, Flatten, Dense

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import SGD, Adam

import pickle

import configuration as cf

import Normalization as Nor
# file configuration.py represent configuration of model

WIDTH = 32

HEIGHT = 32

DEPTH = 3

DIR_TEST_IMAGE = "dataset/image_test"

DIR_TEST_LABEL = "dataset/label_test"



DIR_TRAIN_IMAGE = "dataset/DATASET_IMAGE_64_ALL"

DIR_TRAIN_LABEL = "dataset/DATASET_LABEL_64_ALL"



DIR_DATA_IMAGE = "dataset/DATA_IMAGE"

DIR_DATA_LABEL = "dataset/DATA_LABEL"



FILE_RESULT = "result/RESULTS_32_ALL"

FILE_JSON_SAVE = "save_model/model_32_ALL.json"

FILE_WEIGHT_SAVE = "save_model/model_32_ALL.h5"
# file Normalization.py represent:

# how to load data, generative data, normalize data

import tensorflow as tf 

import cv2

import os

import numpy as np 

import pickle

# import configuration as cf
# from an image we can create K (k = 20) image. It is created for Zoom an brightness

def Zoom(frame):

    w = n_widths + int (n_widths/3)

    h = n_heights + int (n_heights/3)

    image = cv2.resize(frame,(w, h))

    u = int (w/2) - int (n_widths/2)

    v = int (h/2) - int (n_heights/2)

    img_crop = image[v:n_heights+v, u:n_widths+u]

    return img_crop



def increase_brightness(img, value):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # print (v)



    lim = 255 - value

    v[v > lim] = 255

    v[v <= lim] += value



    final_hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img



def generative(x_train ,y_train, image, labels):

	value = 0

	for i in range(10):

		# value = 10

		images = increase_brightness(image, value)

		x_train.append(images)

		y_train.append(int(labels))

		# zoomx

		img_zoom = Zoom(images)

		x_train.append(img_zoom)

		y_train.append(int(labels))

		value += 5
def imread_image(x_train, y_train,paths, labels, id_image):

	image = cv2.imread(paths, cv2.COLOR_GRAY2RGB)

	image = cv2.resize(image, (n_widths, n_heights))

	x_train.append(image)

	y_train.append(int(labels))





def read_test():

	dataX = pickle.load(open(cf.DIR_DATA_IMAGE, "rb"))

	dataY = pickle.load(open(cf.DIR_DATA_LABEL, "rb"))

	return dataX, dataY



def read_train():

	dataX = pickle.load(open(cf.DIR_DATA_IMAGE, "rb"))

	dataY = pickle.load(open(cf.DIR_DATA_LABEL, "rb"))

	x_train = []

	y_train = []

	for i in range(dataX.shape[0]):

		generative(x_train, y_train, dataX[i], dataY[i])

	x_train = np.asarray(x_train)

	y_train = np.asarray(y_train)

	return x_train, y_train



def write_file_test():

	x_test = []

	y_test = []

	director_paths = os.path.abspath(dir_test)

	list_image = os.listdir(director_paths)

	for image in list_image:

		paths = director_paths + '/' + image

		id_image = image

		labels = 100

		imread_image(x_test, y_test, paths, labels, id_image)

	x_test = np.asarray(x_test)

	y_test = np.asarray(y_test)

	print (x_test.shape)

	print (y_test.shape)

	pickle.dump(x_test, open(cf.DIR_TEST_IMAGE, "wb"))

	pickle.dump(y_test, open(cf.DIR_TEST_LABEL, "wb"))



# write_file_test()



def write_file_train():

	x_train = []

	y_train = []

	# count = 0

	# tong = 0

	director_paths = os.path.abspath(dir_train)

	list_file = os.listdir(director_paths)

	for file in list_file:

		file_paths = director_paths + '/' + file

		list_image = os.listdir(file_paths)

		print ("file = ", file, " size of file: ", len(list_image))

		for i in range(0, len(list_image)):

			id_image = list_image[i]

			paths = file_paths + '/' + id_image

			imread_image(x_train, y_train, paths, file, id_image)

	x_train = np.asarray(x_train)

	y_train = np.asarray(y_train)

	print (x_train.shape)

	print (y_train.shape)

	pickle.dump(x_train, open(cf.DIR_DATA_IMAGE, "wb"))

	pickle.dump(y_train, open(cf.DIR_DATA_LABEL, "wb"))



# write_file_train()
def normalize():

	dataX, y_train = read_train()

	# dataX = pickle.load(open(cf.DIR_TRAIN_IMAGE, "rb"))

	# dataY = pickle.load(open(cf.DIR_TEST_IMAGE, "rb"))

	dataY, y__ = read_test()

	dataX = dataX.astype(np.float32)

	dataY = dataY.astype(np.float32)

	print (dataX.shape)

	print (dataY.shape)

	print ("start")

	n = dataX.shape[1]*dataX.shape[2]*dataX.shape[3]

	x_train = dataX.reshape((dataX.shape[0], dataX.shape[1]*dataX.shape[2]*dataX.shape[3]))

	x_test = dataY.reshape((dataY.shape[0], dataY.shape[1]*dataY.shape[2]*dataY.shape[3]))

	max_x = np.max(x_train, axis = 0)

	min_x = np.min(x_train, axis = 0)

	avagen = np.sum(x_train, axis = 0) * 1.0 / x_train.shape[0]

	print ("Loaded finsh")

	for i in  range(n):

		r = max_x[i] - min_x[i]

		if r==0:

			r = 1

		# x_train[:, i:i+1] = (x_train[: , i:i+1] - avagen[i])/r

		x_test[:, i:i+1] = (x_test[: , i:i+1] - avagen[i])/r

	# x_train = x_train.reshape((x_train.shape[0], cf.WIDTH, cf.HEIGHT, cf.DEPTH))

	x_test = x_test.reshape((x_test.shape[0], cf.WIDTH, cf.HEIGHT, cf.DEPTH))

	return x_test
# build model network CNN

def build_model_CNN():

	inputs = Input(shape = (cf.WIDTH, cf.HEIGHT, cf.DEPTH), name = 'input')



	layers = inputs

	i = 0



	for filters in n_neurons:

		layers = Conv2D(filters = filters, kernel_size = (2,2), strides = (1,1), 

						padding = "SAME", activation = 'relu')(layers)



		layers = MaxPool2D(pool_size = (2,2), strides = (2,2))(layers)

		layers = Dropout(0.02)(layers)

		i += 1

	layers = Flatten()(layers)



	layers = Dense(units = 512, activation = 'relu')(layers)

	layers = Dropout(0.02)(layers)



	# layers = Dense(units = 1024, activation = 'relu')(layers)

	# layers = Dropout(0.02)(layers)



	outputs = Dense(units = n_outputs, activation = 'softmax')(layers)



	model = Model(inputs = inputs, outputs = outputs)



	model.summary()

	# serialize model to Json

	model_json = model.to_json()

	with open(cf.FILE_JSON_SAVE, "w") as json_file:

		json_file.write(model_json)



	# serialize weight to HDF5

	model.save_weights(cf.FILE_WEIGHT_SAVE)

	print ("saved model to disk")



build_model_CNN()
import tensorflow as tf 

import numpy as np 

from tensorflow.keras.models import Model, model_from_json

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.layers import Dense, Input, Dropout

import pickle

import Read_Write_CSV as RW

import csv

import cv2

import configuration as cf

import Normalization as Nor



# define hyper-parammeter

learning_rate = 0.001

x_test, y_test = Nor.read_test()

# print (y_test[:100])

x_test = Nor.normalize()





def build_model():

	# load json and create model

	json_file = open(cf.FILE_JSON_SAVE, "r")

	json_file_loaded = json_file.read()

	json_file.close()



	# load model

	model = model_from_json(json_file_loaded)

	model.load_weights(cf.FILE_WEIGHT_SAVE)

	print ("loaded model from disk")



	model.compile(loss = 'categorical_crossentropy', 

					optimizer = Adam(learning_rate),

					metrics = ['accuracy'])

	prediction = model.predict(x_test)

	predict = np.argmax(prediction, 1)

	pickle.dump(predict, open(cf.DIR_DATA_LABEL, "wb"))

build_model()	

# file CNN_Keras.py

import tensorflow as tf 

import numpy as np 

from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPool2D, Flatten, Dense

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import SGD, Adam

import pickle

import configuration as cf

import Normalization as Nor



# define hyper-parammeter

n_neurons = [128,256,512, 512, 1024]

n_layers = len(n_neurons)

n_outputs = 43



# normalize data

def normalize(dataX):

	print ("start")

	n = dataX.shape[1]*dataX.shape[2]*dataX.shape[3]

	x_train = dataX.reshape((dataX.shape[0], dataX.shape[1]*dataX.shape[2]*dataX.shape[3]))

	max_x = np.max(x_train, axis = 0)

	min_x = np.min(x_train, axis = 0)

	avagen = np.sum(x_train, axis = 0) * 1.0 / x_train.shape[0]

	print ("Loaded finsh")

	for i in  range(n):

		r = max_x[i] - min_x[i]

		if r==0:

			r = 1

		x_train[:, i:i+1] = (x_train[: , i:i+1] - avagen[i])/r

	x_train = x_train.reshape((x_train.shape[0], cf.WIDTH, cf.HEIGHT, cf.DEPTH))

	return x_train



x_train, y_train = Nor.read_train()

x_train = x_train.astype(np.float32)

y_train = tf.keras.utils.to_categorical(y_train, n_outputs)

x_train = normalize(x_train)



# build model network CNN

def build_model_CNN():

	inputs = Input(shape = (cf.WIDTH, cf.HEIGHT, cf.DEPTH), name = 'input')



	layers = inputs

	i = 0



	for filters in n_neurons:

		layers = Conv2D(filters = filters, kernel_size = (2,2), strides = (1,1), 

						padding = "SAME", activation = 'relu')(layers)



		layers = MaxPool2D(pool_size = (2,2), strides = (2,2))(layers)

		layers = Dropout(0.02)(layers)

		i += 1

	layers = Flatten()(layers)



	layers = Dense(units = 512, activation = 'relu')(layers)

	layers = Dropout(0.02)(layers)



	outputs = Dense(units = n_outputs, activation = 'softmax')(layers)



	model = Model(inputs = inputs, outputs = outputs)



	model.summary()

	model.compile(loss = 'categorical_crossentropy', 

					optimizer = Adam(0.001), 

					metrics = ['accuracy'])



	model.fit(x_train, y_train, batch_size = 100, epochs = 3)



	# serialize model to Json

	model_json = model.to_json()

	with open(cf.FILE_JSON_SAVE, "w") as json_file:

		json_file.write(model_json)



	# serialize weight to HDF5

	model.save_weights(cf.FILE_WEIGHT_SAVE)

	print ("saved model to disk")



build_model_CNN()
import tensorflow as tf 

import numpy as np 

from tensorflow.keras.models import Model, model_from_json

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.layers import Dense, Input, Dropout

import pickle

import Read_Write_CSV as RW

import csv

import cv2

import configuration as cf

import Normalization as Nor

import cv2



# define hyper-parammeter

learning_rate = 0.001



def load_data():

	x_test = pickle.load(open(cf.DIR_TEST_IMAGE, "rb"))

	y_test_all = pickle.load(open(cf.DIR_TEST_LABEL, "rb"))

	

	y_test = y_test_all[:, 1:]

	y_id_name = y_test_all[:, :1]



	x_test = x_test.astype(np.float32)



	return x_test, y_test, y_id_name



def build_model():



	print ("start")

	x_test, y_test, y_id_name = load_data()

	id_ = y_id_name.tolist()

	results = RW.read_csv()



	x_test_2, y_test_2 = Nor.read_test()

	x_display = x_test_2

	x_test_2, x_test = Nor.data_test_noisy()



	# load json and create model

	json_file = open(cf.FILE_JSON_SAVE, "r")

	json_file_loaded = json_file.read()

	json_file.close()



	# load model

	model = model_from_json(json_file_loaded)

	model.load_weights(cf.FILE_WEIGHT_SAVE)

	print ("loaded model from disk")



	model.compile(loss = 'categorical_crossentropy', 

					optimizer = Adam(learning_rate),

					metrics = ['accuracy'])



	pre_prob = model.predict_proba(x_test_2)



	print ("predict data test")

	prediction = model.predict(x_test)

	predict = np.argmax(prediction, 1)

	sorce = predict.tolist()

	print ("finsh prediction")

	for i in range(len(sorce)):

		id_name = id_[i][0]

		results[id_name] = sorce[i]

		# print ("id = ", id_name, ":  ", sorce[i])

	pickle.dump(results, open(cf.FILE_RESULT, "wb"))

	print ("write successful")

build_model()	
