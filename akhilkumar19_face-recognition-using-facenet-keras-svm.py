# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# example of loading the keras facenet model

from keras.models import load_model

# load the model

model = load_model('/kaggle/input/facenet-keras/facenet_keras.h5')

# summarize input and output shape

print(model.inputs)

print(model.outputs)
!pip install /kaggle/input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl


# confirm mtcnn was installed correctly

import mtcnn

# print version

print(mtcnn.__version__)
# function for face detection with mtcnn

from PIL import Image

from numpy import asarray

from mtcnn.mtcnn import MTCNN



# extract a single face from a given photograph

def extract_face(filename, required_size=(160, 160)):

	# load image from file

	image = Image.open('/kaggle/input/sample-face-images/00000008.jpg')

	# convert to RGB, if needed

	image = image.convert('RGB')

	# convert to array

	pixels = asarray(image)

	# create the detector, using default weights

	detector = MTCNN()

	# detect faces in the image

	results = detector.detect_faces(pixels)

	# extract the bounding box from the first face

	x1, y1, width, height = results[0]['box']

	# bug fix

	x1, y1 = abs(x1), abs(y1)

	x2, y2 = x1 + width, y1 + height

	# extract the face

	face = pixels[y1:y2, x1:x2]

	# resize pixels to the model size

	image = Image.fromarray(face)

	image = image.resize(required_size)

	face_array = asarray(image)

	return face_array



# load the photo and extract the face

pixels = extract_face('...')

print(pixels)
# demonstrate face detection on 5 Celebrity Faces Dataset

from os import listdir

from PIL import Image

from numpy import asarray

from matplotlib import pyplot

from mtcnn.mtcnn import MTCNN



# extract a single face from a given photograph

def extract_face(filename, required_size=(160, 160)):

	# load image from file

	image = Image.open(filename)

	# convert to RGB, if needed

	image = image.convert('RGB')

	# convert to array

	pixels = asarray(image)

	# create the detector, using default weights

	detector = MTCNN()

	# detect faces in the image

	results = detector.detect_faces(pixels)

	# extract the bounding box from the first face

	x1, y1, width, height = results[0]['box']

	# bug fix

	x1, y1 = abs(x1), abs(y1)

	x2, y2 = x1 + width, y1 + height

	# extract the face

	face = pixels[y1:y2, x1:x2]

	# resize pixels to the model size

	image = Image.fromarray(face)

	image = image.resize(required_size)

	face_array = asarray(image)

	return face_array



# specify folder to plot

folder = '/kaggle/input/5-celebrity-faces-dataset/train/ben_afflek/'

i = 1

# enumerate files

for filename in listdir(folder):

	# path

	path = folder + filename

	# get face

	face = extract_face(path)

	print(i, face.shape)

	# plot

	pyplot.subplot(2, 7, i)

	pyplot.axis('off')

	pyplot.imshow(face)

	i += 1

pyplot.show()
# load images and extract faces for all images in a directory

def load_faces(directory):

	faces = list()

	# enumerate files

	for filename in listdir(directory):

		# path

		path = directory + filename

		# get face

		face = extract_face(path)

		# store

		faces.append(face)

	return faces
# load a dataset that contains one subdir for each class that in turn contains images

def load_dataset(directory):

	X, y = list(), list()

	# enumerate folders, on per class

	for subdir in listdir(directory):

		# path

		path = directory + subdir + '/'

		# skip any files that might be in the dir

		#

		# load all faces in the subdirectory

		faces = load_faces(path)

		# create labels

		labels = [subdir for _ in range(len(faces))]

		# summarize progress

		print('>loaded %d examples for class: %s' % (len(faces), subdir))

		# store

		X.extend(faces)

		y.extend(labels)

	return asarray(X), asarray(y)
# load train dataset

trainX, trainy = load_dataset('/kaggle/input/5-celebrity-faces-dataset/train/')

print(trainX.shape, trainy.shape)

# load test dataset

testX, testy = load_dataset('/kaggle/input/5-celebrity-faces-dataset/val/')

print(testX.shape, testy.shape)

# save arrays to one file in compressed format

import numpy

numpy.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
# face detection for the 5 Celebrity Faces Dataset

from os import listdir

from os.path import isdir

from PIL import Image

from matplotlib import pyplot

from numpy import savez_compressed

from numpy import asarray

import numpy

from mtcnn.mtcnn import MTCNN



# extract a single face from a given photograph

def extract_face(filename, required_size=(160, 160)):

	# load image from file

	image = Image.open(filename)

	# convert to RGB, if needed

	image = image.convert('RGB')

	# convert to array

	pixels = asarray(image)

	# create the detector, using default weights

	detector = MTCNN()

	# detect faces in the image

	results = detector.detect_faces(pixels)

	# extract the bounding box from the first face

	x1, y1, width, height = results[0]['box']

	# bug fix

	x1, y1 = abs(x1), abs(y1)

	x2, y2 = x1 + width, y1 + height

	# extract the face

	face = pixels[y1:y2, x1:x2]

	# resize pixels to the model size

	image = Image.fromarray(face)

	image = image.resize(required_size)

	face_array = asarray(image)

	return face_array



# load images and extract faces for all images in a directory

def load_faces(directory):

	faces = list()

	# enumerate files

	for filename in listdir(directory):

		# path

		path = directory + filename

		# get face

		face = extract_face(path)

		# store

		faces.append(face)

	return faces



# load a dataset that contains one subdir for each class that in turn contains images

def load_dataset(directory):

	X, y = list(), list()

	# enumerate folders, on per class

	for subdir in listdir(directory):

		# path

		path = directory + subdir + '/'

		# skip any files that might be in the dir

		#is dir removed here ######################

		# load all faces in the subdirectory

		faces = load_faces(path)

		# create labels

		labels = [subdir for _ in range(len(faces))]

		# summarize progress

		print('>loaded %d examples for class: %s' % (len(faces), subdir))

		# store

		X.extend(faces)

		y.extend(labels)

	return asarray(X), asarray(y)



# load train dataset

trainX, trainy = load_dataset('/kaggle/input/5-celebrity-faces-dataset/train/')

print(trainX.shape, trainy.shape)

# load test dataset

testX, testy = load_dataset('/kaggle/input/5-celebrity-faces-dataset/val/')

# save arrays to one file in compressed format

numpy.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
# load the face dataset

import numpy

data = numpy.load('5-celebrity-faces-dataset.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the facenet model

import keras.models

model = keras.models.load_model('/kaggle/input/facenet-keras/facenet_keras.h5')

print('Loaded Model')
from numpy import load

from numpy import expand_dims

from numpy import asarray

from numpy import savez_compressed

from keras.models import load_model

def get_embedding(model, face_pixels):

	# scale pixel values

	face_pixels = face_pixels.astype('float32')

	# standardize pixel values across channels (global)

	mean, std = face_pixels.mean(), face_pixels.std()

	face_pixels = (face_pixels - mean) / std

	# transform face into one sample

	samples = expand_dims(face_pixels, axis=0)

	# make prediction to get embedding

	yhat = model.predict(samples)

	return yhat[0]

newTrainX = list()

for face_pixels in trainX:

	embedding = get_embedding(model, face_pixels)

	newTrainX.append(embedding)

newTrainX = asarray(newTrainX)

print(newTrainX.shape)

# convert each face in the test set to an embedding

newTestX = list()

for face_pixels in testX:

	embedding = get_embedding(model, face_pixels)

	newTestX.append(embedding)

newTestX = asarray(newTestX)

print(newTestX.shape)
numpy.savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
# calculate a face embedding for each face in the dataset using facenet

from numpy import load

from numpy import expand_dims

from numpy import asarray

from numpy import savez_compressed

from keras.models import load_model



# get the face embedding for one face

def get_embedding(model, face_pixels):

	# scale pixel values

	face_pixels = face_pixels.astype('float32')

	# standardize pixel values across channels (global)

	mean, std = face_pixels.mean(), face_pixels.std()

	face_pixels = (face_pixels - mean) / std

	# transform face into one sample

	samples = expand_dims(face_pixels, axis=0)

	# make prediction to get embedding

	yhat = model.predict(samples)

	return yhat[0]



# load the face dataset

data = load('5-celebrity-faces-dataset.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# load the facenet model

model = load_model('/kaggle/input/facenet-keras/facenet_keras.h5')

print('Loaded Model')

# convert each face in the train set to an embedding

newTrainX = list()

for face_pixels in trainX:

	embedding = get_embedding(model, face_pixels)

	newTrainX.append(embedding)

newTrainX = asarray(newTrainX)

print(newTrainX.shape)

# convert each face in the test set to an embedding

newTestX = list()

for face_pixels in testX:

	embedding = get_embedding(model, face_pixels)

	newTestX.append(embedding)

newTestX = asarray(newTestX)

print(newTestX.shape)

# save arrays to one file in compressed format

savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
from numpy import load

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC

# load dataset

data = load('5-celebrity-faces-embeddings.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors

in_encoder = Normalizer(norm='l2')

trainX = in_encoder.transform(trainX)

testX = in_encoder.transform(testX)
# label encode targets

out_encoder = LabelEncoder()

out_encoder.fit(trainy)

trainy = out_encoder.transform(trainy)

testy = out_encoder.transform(testy)

print(trainy,testy)
# fit model

model = SVC(kernel='linear')

model.fit(trainX, trainy)
# predict

yhat_train = model.predict(trainX)

yhat_test = model.predict(testX)

# score

score_train = accuracy_score(trainy, yhat_train)

score_test = accuracy_score(testy, yhat_test)

# summarize

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
# develop a classifier for the 5 Celebrity Faces Dataset

from numpy import load

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC

# load dataset

data = load('5-celebrity-faces-embeddings.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

# normalize input vectors

in_encoder = Normalizer(norm='l2')

trainX = in_encoder.transform(trainX)

testX = in_encoder.transform(testX)

# label encode targets

out_encoder = LabelEncoder()

out_encoder.fit(trainy)

trainy = out_encoder.transform(trainy)

testy = out_encoder.transform(testy)

# fit model

model = SVC(kernel='linear', probability=True)

model.fit(trainX, trainy)

# predict

yhat_train = model.predict(trainX)

yhat_test = model.predict(testX)

# score

score_train = accuracy_score(trainy, yhat_train)

score_test = accuracy_score(testy, yhat_test)

# summarize

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
# load faces

data = load('5-celebrity-faces-dataset.npz')

testX_faces = data['arr_2']
from random import choice

# test model on a random example from the test dataset

selection = choice([i for i in range(testX.shape[0])])

random_face_pixels = testX_faces[selection]

random_face_emb = testX[selection]

random_face_class = testy[selection]

random_face_name = out_encoder.inverse_transform([random_face_class])

print(random_face_name)
# prediction for the face

samples = expand_dims(random_face_emb, axis=0)

yhat_class = model.predict(samples)

yhat_prob = model.predict_proba(samples)
# get name

class_index = yhat_class[0]

class_probability = yhat_prob[0,class_index] * 100

predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

print('Expected: %s' % random_face_name[0])
# plot for fun

pyplot.imshow(random_face_pixels)

title = '%s (%.3f)' % (predict_names[0], class_probability)

pyplot.title(title)

pyplot.show()
# develop a classifier for the 5 Celebrity Faces Dataset

from random import choice

from numpy import load

from numpy import expand_dims

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC

from matplotlib import pyplot

# load faces

data = load('5-celebrity-faces-dataset.npz')

testX_faces = data['arr_2']

# load face embeddings

data = load('5-celebrity-faces-embeddings.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors

in_encoder = Normalizer(norm='l2')

trainX = in_encoder.transform(trainX)

testX = in_encoder.transform(testX)

# label encode targets

out_encoder = LabelEncoder()

out_encoder.fit(trainy)

trainy = out_encoder.transform(trainy)

testy = out_encoder.transform(testy)

# fit model

model = SVC(kernel='linear', probability=True)

model.fit(trainX, trainy)

# test model on a random example from the test dataset

selection = choice([i for i in range(testX.shape[0])])

random_face_pixels = testX_faces[selection]

random_face_emb = testX[selection]

random_face_class = testy[selection]

random_face_name = out_encoder.inverse_transform([random_face_class])

# prediction for the face

samples = expand_dims(random_face_emb, axis=0)

yhat_class = model.predict(samples)

yhat_prob = model.predict_proba(samples)

# get name

class_index = yhat_class[0]

class_probability = yhat_prob[0,class_index] * 100

predict_names = out_encoder.inverse_transform(yhat_class)

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

print('Expected: %s' % random_face_name[0])

# plot for fun

pyplot.imshow(random_face_pixels)

title = '%s (%.3f)' % (predict_names[0], class_probability)

pyplot.title(title)

pyplot.show()