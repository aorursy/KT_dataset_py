import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from PIL import Image
image = Image.open('/kaggle/input/5-celebrity-faces-dataset/data/train/elton_john/httpwwwlautdeEltonJohneltonjohnjpg.jpg')



image = image.convert('RGB')



plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')

plt.show()
from numpy import asarray

# convert to array

pixels = asarray(image)
!pip install mtcnn
# confirm mtcnn was installed correctly

import mtcnn

# print version

print(mtcnn.__version__)
from mtcnn.mtcnn import MTCNN

# create the detector, using default weights

detector = MTCNN()
# detect faces in the image

results = detector.detect_faces(pixels)
# extract the bounding box from the first face

x1, y1, width, height = results[0]['box']

x2, y2 = x1 + width, y1 + height
# extract the face

face = np.array(image)[y1:y2, x1:x2]
plt.imshow(face)
from PIL import Image

# resize pixels to the model size

image = Image.fromarray(face)

image = image.resize((160, 160))

face_array = np.asarray(image)
plt.imshow(face_array)
from keras.models import load_model

model = load_model('/kaggle/input/facenet/facenet_keras.h5')

print('Loaded Model')

print(model.inputs)

print(model.outputs)
import os



def extract_face(filename, required_size=(160, 160)):

    # load image from file

    image = Image.open(filename)

    # convert to RGB, if needed

    image = image.convert('RGB')

    # convert to array

    pixels = np.asarray(image)

    # create the detector, using default weights

    detector = MTCNN()

    # detect faces in the image

    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face

    x1, y1, width, height = results[0]['box']

    # deal with negative pixel index

    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1 + width, y1 + height

    # extract the face

    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size

    image = Image.fromarray(face)

    image = image.resize(required_size)

    face_array = np.asarray(image)

    return face_array



def load_face(dir):

    faces = list()

    # enumerate files

    for filename in os.listdir(dir):

        path = dir + filename

        face = extract_face(path)

        faces.append(face)

    return faces



def load_dataset(dir):

    # list for faces and labels

    X, y = list(), list()

    for subdir in os.listdir(dir):

        path = dir + subdir + '/'

        faces = load_face(path)

        labels = [subdir for i in range(len(faces))]

        print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress

        X.extend(faces)

        y.extend(labels)

    return np.asarray(X), np.asarray(y)





# load train dataset

trainX, trainy = load_dataset('/kaggle/input/5-celebrity-faces-dataset/data/train/')

print(trainX.shape, trainy.shape)

# load test dataset

testX, testy = load_dataset('/kaggle/input/5-celebrity-faces-dataset/data/val/')

print(testX.shape, testy.shape)



from numpy import expand_dims



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
df = pd.DataFrame(newTrainX)

df["target"] = trainy

df.head()
import seaborn as sns

from sklearn.decomposition import PCA

# Create a PCA instance:

pca = PCA(n_components=2) 

# Fit pca to 'X'

pca_features = pca.fit_transform(newTrainX)

print (pca_features.shape)



df_plot = pd.DataFrame(pca_features)

df_plot["target"] = trainy



plt.figure(figsize=(16, 6))

sns.scatterplot(x=df_plot[0] , y= df_plot[1], data = df_plot,  hue = "target" )
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer
newTrainX.shape , newTestX.shape , testy.shape



normaliser = Normalizer()

emdTrainX_norm = normaliser.fit_transform(newTrainX)

emdTestX_norm = normaliser.transform(newTestX)



encoder = LabelEncoder()

trainy_enc = encoder.fit_transform(trainy)

testy_enc = encoder.transform(testy)
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

model = SVC(kernel='linear', probability=True)

model.fit(emdTrainX_norm, trainy_enc)

# predict

yhat_train = model.predict(emdTrainX_norm)

yhat_test = model.predict(emdTestX_norm)

# score

score_train = accuracy_score(trainy_enc, yhat_train)

score_test = accuracy_score(testy_enc, yhat_test)

# summarize

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
from random import choice

# select a random face from test set

selection = choice([i for i in range(testX.shape[0])])

random_face = testX[selection]

random_face_emd = emdTestX_norm[selection]

random_face_class = testy_enc[selection]

random_face_name = encoder.inverse_transform([random_face_class])



# prediction for the face

samples = np.expand_dims(random_face_emd, axis=0)

yhat_class = model.predict(samples)

yhat_prob = model.predict_proba(samples)

# get name

class_index = yhat_class[0]

class_probability = yhat_prob[0,class_index] * 100

predict_names = encoder.inverse_transform(yhat_class)

all_names = encoder.inverse_transform([0,1,2,3,4])

#print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0]*100))

print('Expected: %s' % random_face_name[0])

# plot face

plt.imshow(random_face)

title = '%s (%.3f)' % (predict_names[0], class_probability)

plt.title(title)

plt.show()