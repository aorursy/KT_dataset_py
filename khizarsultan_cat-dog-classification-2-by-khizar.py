from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D, BatchNormalization,Dropout

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import load_model

#Prediction of test set

from sklearn.metrics import confusion_matrix 

import seaborn as sns



# from google.colab import files

import zipfile

import tensorflow as tf

import os





%matplotlib inline

import pandas as pd



from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np



# predicting the images

# from urls

from PIL import Image

import requests

import cv2

from io import BytesIO
classifier = load_model('../models/cat_dog_96.h5')
#Prediction of image

def predict_image(img_type):

    img1 = image.load_img(img_type, target_size=(64, 64))

    img = image.img_to_array(img1)

    img_norm = img/255

    img_4d = np.expand_dims(img_norm, axis=0)

    prediction = classifier.predict_classes(img_4d, batch_size=None)

    if prediction[0][0] == 1:

        print("It is Dog")

    elif prediction[0][0] == 0:

        print("It is Cat")

    plt.imshow(img1)
one_cat_image = "../data/cat-dog/validate/cat/cat.100.jpg"

one_dog_image = "../data/cat-dog/validate/dog/dog.1.jpg"
predict_image(one_cat_image)
predict_image(one_dog_image)
# test_set.reset

#Validation Set

batch_size = 32

IMG_WIDTH,IMG_HEIGHT = 64,64

epoches = 200



test_datagen = ImageDataGenerator(rescale=1./255)



test_set = test_datagen.flow_from_directory('../data/cat-dog/validate',

                                           target_size=(IMG_WIDTH,IMG_HEIGHT),

                                           batch_size = batch_size,

                                           class_mode='binary',

                                           shuffle=False)



test_set1 = test_datagen.flow_from_directory('../data/cat-dog/test',

                                           target_size=(IMG_WIDTH,IMG_HEIGHT),

                                           batch_size = batch_size,

                                           shuffle=False)

ytesthat = classifier.predict_classes(test_set, batch_size=None)
df = pd.DataFrame({

    'filename':test_set.filenames,

    'Actual':test_set.classes,

    'predict':ytesthat[:,0],

})

# pd.set_option('display.float_format', lambda x: '%.5f' % x)

# df['y_pred'] = df['predict'] > 0.5

# df.y_pred = df.y_pred.astype(int)

# df

df.head(10)
missclassify = df[df.Actual != df.predict]
print(f"Total misclassified image from 5000 Validation images are : {missclassify['Actual'].count()}")
conf_matrix = confusion_matrix(df.Actual,df.predict)
sns.heatmap(conf_matrix,annot=True,fmt='g');

plt.xlabel('predicted value')

plt.ylabel('true value');
#  for generator image set u can use 

# ypred = classifier.predict_generator(test_set)



fig=plt.figure(figsize=(20, 10))

columns = 7

rows = 4

for i in range(columns*rows):

    fig.add_subplot(rows, columns, i+1)

    

    path = "../data/cat-dog/test/"+test_set1.filenames[i].replace('\\','/')

    img1 = image.load_img(path, target_size=(64, 64))

    

#     img1 = image.load_img('test1/'+test_set1.filenames[np.random.choice(range(12500))], target_size=(64, 64))

    img = image.img_to_array(img1)

    img = img/255

    img = np.expand_dims(img, axis=0)

    prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.

    if(prediction[:,:]>0.5):

        value ='Dog :%1.2f'%(prediction[0,0])

        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))

    else:

        value ='Cat :%1.2f'%(1.0-prediction[0,0])

        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))

    plt.imshow(img1)
