!pip install mtcnn

from mtcnn import MTCNN

import os

from PIL import Image

from numpy import asarray

detector = MTCNN()



d = '/kaggle/input/images/'

for subdir in os.listdir(d):

    path = d + subdir + '/'

    for filename in os.listdir(path):

        try:

            image = Image.open(path+filename)     

            image = image.convert('RGB')

            pixels = asarray(image)

            r = detector.detect_faces(pixels)

            x1, y1, width, height = r[0]['box']

            print('Face detected, moving along {}'.format(path+filename))

        except:

            print('No face detected, removing {}'.format(path+filename))

            os.remove(path+filename)

        finally:

            image.close()

print('Done')
!pip install split_folders

print('Splitting into dateset/train and dataset/val...')

import split_folders

split_folders.ratio('/kaggle/input/images/','/kaggle/output/dataset/',ratio=(.7, .3))

print('Done')
!pip install mtcnn

from mtcnn.mtcnn import  MTCNN

from PIL import Image

from numpy import asarray, savez_compressed, load, expand_dims

from keras.models import load_model

import os



classes = os.listdir('/kaggle/output/dataset/train/')

print('Total classes: {} \n {}'.format(len(classes),classes))



detector = MTCNN()



# extract face bounding box from image

def extract_face(filename,required_size=(160,160)):

    image = Image.open(filename)

    image = image.convert('RGB')

    pixels = asarray(image) 

    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']

    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)

    image = image.resize(required_size)

    face_array = asarray(image)

    image.close()

    return face_array



# load face bounding box from image

def load_faces(directory):

    faces = list()

    #enumerate files

    for filename in os.listdir(directory):

        path = directory + filename 

        face = extract_face(path)

        faces.append(face)

    return faces



# load a dataset that contains one subdir for each class that in turn contains images

def load_dataset(directory):

    x, y = list(), list()

    for subdir in os.listdir(directory):

        path = directory +'/' + subdir + '/'

        faces = load_faces(path) #load all cropped faces in the subdir

        labels = [subdir for _ in range(len(faces))] #create labels

        print('Load {} examples for class: {}'.format(len(faces), subdir))

        x.extend(faces)

        y.extend(labels)

    return asarray(x), asarray(y)



# create face embedding

def get_embedding(model, face_pixels):

    face_pixels = face_pixels.astype('float32')

    #normalize pixel value across channel

    mean, std = face_pixels.mean(), face_pixels.std()

    face_pixels = (face_pixels - mean) / std

    #transform face into one sample

    samples = expand_dims(face_pixels, axis=0)

    #make prediction to get embedding | yhat is a vector

    yhat = model.predict(samples)

    return yhat[0]



# load train dataset

print('Loading train set')

trainX, trainY = load_dataset('/kaggle/output/dataset/train/')



# load test dataset

print('Loading test set')

testX, testY = load_dataset('/kaggle/output/dataset/val')



savez_compressed('faces_dataset.npz', trainX, trainY, testX, testY) #save to npz

print('Saved faces_dataset.npz')



# load pre-trained model

model = load_model('/kaggle/input/model/facenet_keras.h5')

print('Loaded Facenet model')



# convert each face in the train set to an embedding

print('Vetorizing train set')

newTrainX = list()

for face_pixels in trainX:

    embedding = get_embedding(model,face_pixels)

    newTrainX.append(embedding)

newTrainX = asarray(newTrainX)

print(newTrainX.shape)



# convert each face in the test to an embedding

print('Vetorizing test set')

newTestX = list()

for face_pixels in testX:

    embedding = get_embedding(model,face_pixels)

    newTestX.append(embedding)

newTestX = asarray(newTestX)

print(newTestX.shape)

savez_compressed('faces_dataset_embedding.npz', newTrainX, trainY, newTestX, testY)

print('Saved faces_dataset_embedding.npz')

print('Done')
from numpy import load

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC

# load dataset

data = load('faces_dataset_embedding.npz')

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
# load facenet model

faceNetModel = load_model('/kaggle/input/model/facenet_keras.h5')

from PIL import Image

import requests



def get_face_from_url(url):

    image = Image.open(requests.get(url, stream=True).raw)

    pixels = asarray(image)

    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']

    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)

    image = image.resize((160,160))

    pixels = asarray(image)

    face_emb = get_embedding(faceNetModel, pixels)

    return image, face_emb
url = 'https://vcdn1-giaitri.vnecdn.net/2018/12/04/nhiet-ba-1543906981.jpg?w=1200&h=0&q=100&dpr=1&fit=crop&s=K1-afDhPFAYDOezjBhtUaA'

image, face_emb = get_face_from_url(url)

samples = expand_dims(face_emb, axis=0)

yhat_class = model.predict(samples)

yhat_prob = model.predict_proba(samples)

image
class_index = yhat_class[0]

class_probability = yhat_prob[0,class_index] * 100

predict_names = out_encoder.inverse_transform(yhat_class)

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))