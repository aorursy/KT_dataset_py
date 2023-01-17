# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
#from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
# resize pixels to the model size
    image = Image.fromarray(pixels)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
img = extract_face('/kaggle/input/masked-facerecognition/AFDB_face_dataset/AFDB_face_dataset/aidai/1_0_aidai_0003.jpg')
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
print(img.shape)
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
        # print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)
# load train dataset
trainX, trainy = load_dataset('/kaggle/input/masked-facerecognition/AFDB_face_dataset/AFDB_face_dataset/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('/kaggle/input/masked-facerecognition/AFDB_masked_face_dataset/AFDB_masked_face_dataset/')
print(testX.shape, testy.shape)
# save and compress the dataset for further use
np.savez_compressed('new_maskes_face.npz', trainX, trainy, testX, testy)
data = np.load('/kaggle/input/new-masked-face/extracted_masked_unmasked.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
trainx, valid = train_test_split(trainX, test_size=0.1, random_state=42, shuffle=True)
print("number of image in train dataset : %s" %(len(trainx)))

print("number of image in train dataset : %s" %(len(valid)))
y_train, y_valid = train_test_split(trainy, test_size=0.1, random_state=42, shuffle=True)
print("number of image in train dataset : %s" %(len(y_train)))

print("number of image in train dataset : %s" %(len(y_valid)))
# save and compress the dataset for further use
np.savez_compressed('extracted_masked_unmasked.npz', trainx, y_train, valid, y_valid,testX, testy)
data = np.load('/kaggle/input/new-masked-face/extracted_masked_unmasked.npz')
trainx, y_train, valid, y_valid,testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5']
print('Loaded: ', trainx.shape, y_train.shape, valid.shape, y_valid.shape,testX.shape, testy.shape)
facenet_model = load_model('/kaggle/input/facenet/keras-facenet/model/facenet_keras.h5')
print('Loaded Model')
print('Loaded: ', trainx.shape, y_train.shape, valid.shape, y_valid.shape,testX.shape, testy.shape)
def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]
emdTrainX = list()
for face in trainx:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)
emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)
embValid = list()
for face in valid:
    emd = get_embedding(facenet_model,face)
    embValid.append(emd)
embValid = np.asarray(embValid)
print(embValid.shape)
emdTestX = list()
for face in testX:
    emd = get_embedding(facenet_model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)
# save arrays to one file in compressed format
np.savez_compressed('embeddings_masked.npz', emdTrainX, y_train, embValid, y_valid, emdTestX, testy)
data = np.load('/kaggle/input/newsst7/embeddings_masked.npz')
emdTrainX, y_train, embValid, y_valid, emdTestX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5']
print('Loaded: ', emdTrainX.shape, y_train.shape, embValid.shape, y_valid.shape, emdTestX.shape, testy.shape)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
print("Dataset: train=%d,validation = %d, test=%d" % (emdTrainX.shape[0],embValid.shape[0] ,emdTestX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
emdTrainX_norm = in_encoder.transform(emdTrainX)
embValid_norm = in_encoder.transform(embValid)
emdTestX_norm = in_encoder.transform(emdTestX)
# label encode targets
out_encoder = LabelEncoder()
encoder_arr = np.append (y_train, 'wangnan')
out_encoder.fit(encoder_arr)
trainy_enc = out_encoder.transform(y_train)
y_valid_enc = out_encoder.transform(y_valid)
testy_enc = out_encoder.transform(testy)
model = SVC(kernel='linear', probability=True)
#model = SVC(kernel='poly', probability=True)
#model = SVC(kernel='rbf', probability=True)
model.fit(emdTrainX_norm, trainy_enc)
# predict
yhat_valid = model.predict(embValid_norm)
yhat_test = model.predict(emdTestX_norm)
# score
score_valid = accuracy_score(y_valid_enc, yhat_valid)
score_test = accuracy_score(testy_enc, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_valid*100, score_test*100))
filename = 'linear.sav'
pickle.dump(model, open(filename, 'wb'))
#filename = 'poly.sav'
#pickle.dump(model, open(filename, 'wb'))
#filename = 'rbf.sav'
#pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open('linear.sav', 'rb'))
from random import choice
for i in range(20):
    # select a random face from test set
    selection = choice([i for i in range(testX.shape[0])]) 
    random_face = testX[selection]
    random_face_emd = emdTestX_norm[selection]
    random_face_class = testy_enc[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = np.expand_dims(random_face_emd, axis=0)
    yhat_class = loaded_model.predict(samples)
    yhat_prob = loaded_model.predict_proba(samples)
    class_index = yhat_class[0]
    if class_index <= 460:
        # get name
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        #if random_face_name[0] == predict_names[0]:
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        print('Expected: %s' % random_face_name[0])
        # plot face
        plt.imshow(random_face)
        title = '%s (%.3f)' % (predict_names[0], class_probability)
        plt.title(title)
        plt.show()