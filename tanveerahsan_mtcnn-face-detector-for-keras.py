import numpy as np

import pandas as pd

import os

import zipfile



print (os.listdir('/kaggle/input'))



#with zipfile.ZipFile('/kaggle/input/15celebrity/14-celebrity-faces-dataset.zip') as z:

#    z.extractall('.')

!pip install mtcnn
import cv2

from matplotlib import pyplot as plt

from mtcnn.mtcnn import MTCNN

from matplotlib.patches import Rectangle

from matplotlib.patches import Circle







def face_extraction(filename,required_size=(160,160)):

    image = cv2.imread(filename)

    shape = image.shape



    if shape[2] == 3:

        pixels = np.asarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    else:

        pixels = np.asarray(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB))



    face_detector = MTCNN()

    results = face_detector.detect_faces(pixels)

    box_image (results,image)

    

    x,y,width,height = results[0]['box']

    x1,y1 = abs(x),abs(y)

    x2,y2 = x1+width, y1+height

    face_detect = image[y1:y2,x1:x2]

    face_detect = np.asarray(face_detect)

    face_array = cv2.resize(face_detect,required_size)

    print (face_array.shape)

    return face_array

def box_image(results,image):

    

    ax = plt.gca()

    plt.imshow(image)

    

    for result in results:

        

        print(result)

        if result['confidence'] > 0.9:

            x,y,width,height = result['box'] 

            rect = Rectangle((x,y),width,height,fill = False, color = 'red')

            ax.add_patch(rect)



    for key,value in result['keypoints'].items():

        circle = Circle(value, radius = 2, color = 'red')

        ax.add_patch(circle)

        

        

    plt.show()

  
def display_image(images):

    plt.imshow(images)

    plt.axis('off')

    plt.show

#image = face_extraction('/kaggle/input/sample-image/25_person_image.JPG')

#display_image(image)

#display_image(image)



image1 = face_extraction('/kaggle/input/sample-image/6_person_image.JPG')

display_image(image1)

image2 = face_extraction('/kaggle/input/sample-image/25_person_image.JPG')

#image3 = face_extraction('/kaggle/input/sample-image/6_person_image_1.JPG')

#image4 = face_extraction('/kaggle/input/sample-image/4_person_image.JPG')

    



display_image(image2)

#display_image(image3)

#display_image(image4)
def load_faces(dir):

    faces = list()

    count=2

    for filename in os.listdir(dir):

        if count > len(faces):

            path = dir + filename

            face = face_extraction(path)

            faces.append(face)

    return faces

    

    

def load_dir(dir):

       

    X, y = list(), list()

    for subdir in os.listdir(dir):

        path = dir + subdir + '/'

        print(path)

        faces = load_faces(path)

        print (len(faces))

        labels = [subdir for i in range(len(faces))]

        print ("Loaded %d samples for class:%s" %(len(faces),subdir)) 

        X.extend(faces)

        y.extend(labels)       

    return np.asarray(X),np.asarray(y)
trainX,trainy = load_dir('/kaggle/input/15celebrity/14-celebrity-faces-dataset/data/train/')

print(trainX.shape,trainy.shape)



testX,testy = load_dir('/kaggle/input/15celebrity/14-celebrity-faces-dataset/data/val/')

print(testX.shape,testy.shape)



np.savez_compressed('14-celebrity-faces.npz',trainX,trainy,testX,testy)
import os



for dir,_,filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dir,filename))
data = np.load ('14-celebrity-faces.npz')

trainX,trainy,testX,testy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

print(trainX.shape,trainy.shape,testX.shape,testy.shape)
from keras.models import load_model



model_filename = '/kaggle/input/facenet-keras/facenet_keras.h5'

facenet_model = load_model(model_filename)

def face_embeddings(model,face):

    face = face.astype('float32')

    mean,std = face.mean(), face.std()

    face = (face-mean) / std

    face_standard = np.expand_dims(face, axis = 0)

    embed = model.predict(face_standard)

    

    return embed[0]
def get_embeddings(faces):

    emb_list = []

    for face in faces :

        emb_output  = face_embeddings(facenet_model,face)

        emb_list.append(emb_output)



    emb_list = np.asarray(emb_list)

    return emb_list



trainX_emb = get_embeddings(trainX)

testX_emb = get_embeddings(testX)

print (trainX.shape,testX.shape)

print (trainX_emb.shape,testX_emb.shape)
np.savez_compressed('14_celebrity_faces_embeddings.npz',trainX_emb,trainy,testX_emb,testy)
data  = np.load('14_celebrity_faces_embeddings.npz')

trainX_emb,trainy,testX_emb,testy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

print(trainX_emb.shape,trainy.shape,testX_emb.shape,testy.shape)
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder,Normalizer

from sklearn.svm import SVC



def data_normalizer(data):

    encoder = Normalizer()

    data_enc = encoder.transform(data)

    return data

    





def recognition_model(model):

    model = SVC(kernel = 'linear', probability = True,random_state = 33)

    return model

    

    

def predict_face_accuracy(input,predict):

       

    acc_score = accuracy_score(input,predict)

    

    

    for input,predict in zip(testY_enc,test_pred):

        if input != predict:

           print (input,"not equals to",predict)

   

    return acc_score

    
trainX,trainY,testX,testY = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

trainX_enc = data_normalizer(trainX)

testX_enc = data_normalizer(testX)

l_encoder = LabelEncoder()

l_encoder.fit(trainY)

trainY_enc = l_encoder.transform(trainY)

testY_enc = l_encoder.transform(testY)

#print(trainY)

print(testY)

#print(trainY_enc)

print(testY_enc)

    
model  = recognition_model(SVC)

model.fit(trainX_enc,trainY_enc)

train_pred = model.predict(trainX_enc)

test_pred = model.predict(testX_enc)
train_acc_score = predict_face_accuracy(trainY_enc,train_pred)

test_acc_score = predict_face_accuracy(testY_enc,test_pred)



print("train_acc_score",train_acc_score)

print("test_acc_score",test_acc_score)
from sklearn.externals import joblib



filename = 'face_detection_model_dump'

joblib.dump(model,filename,compress = 1)
model1 = joblib.load(filename)