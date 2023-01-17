import numpy as np

import pandas as pd

import io

import glob

import base64

from PIL import Image

from keras.preprocessing import image

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Input

from keras.applications import inception_v3

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.optimizers import Adam

from keras.layers import GlobalAveragePooling2D

from keras.layers import Dropout



#STEP 1 Reading Directory Converting to csv/DataFrame

def CreateCsv(path,encoded_category,category):

    img = glob.glob(path)

    img = pd.DataFrame(img)

    img['path'] = img[0]

    img = img.drop([0],axis=1)

    img['label'] = encoded_category

    img['category'] = category

    return(img)

a = CreateCsv('../input/flowers-recognition/flowers/flowers/daisy/*.jpg',0.0,'daisy')

b = CreateCsv('../input/flowers-recognition/flowers/flowers/dandelion/*.jpg',1.0,'dandelion')

c = CreateCsv('../input/flowers-recognition/flowers/flowers/rose/*.jpg',2.0,'rose')

d = CreateCsv('../input/flowers-recognition/flowers/flowers/sunflower/*.jpg',3.0,'sunflower')

e = CreateCsv('../input/flowers-recognition/flowers/flowers/tulip/*.jpg',4.0,'tulip')

images = a.append([b,c,d,e],ignore_index=True)

images.head(5)



im = []

train_image = []

filename = images['path']

def mmeth(length,width):

    

    for filename in images['path']:

        with open(filename, 'rb') as f:

            fname = f.read()

        image_1 = Image.open(io.BytesIO(fname))

        image_1 = image_1.resize((length,width), Image.ANTIALIAS)

        img = image.img_to_array(image_1)

        img = img/255

        train_image.append(img)

    return train_image



train_image= mmeth(200,200)

X = np.array(train_image)

y = images["label"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import keras

y_train_encoded = keras.utils.to_categorical(y_train, num_classes=5, dtype='int32')

y_test_encoded = keras.utils.to_categorical(y_test, num_classes=5, dtype='int32')
y_train_encoded

pd.DataFrame([x for x in np.where(y_train_encoded ==1, y_train_encoded.columns,'').flatten().tolist() if len(x) >0],columns= (["Flower"]) )

import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

import cv2







base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(200, 200, 3))
# Add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(2048, activation='relu')(x)

predictions = Dense(5, activation='softmax')(x)

# The model we will train

model = Model(inputs = base_model.input, outputs = predictions)

# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_encoded, batch_size = 32, epochs = 50,validation_data = (X_test,y_test_encoded))

y_pred = model.predict(X_test)
max(y_pred[1])

for i in y_pred:

    if y_pred[i]<0.5:

        y_pred[i]=0

    elif y_pred[i]>0.5 and y_pred[i]<1.5 :

        y_pred[i]=1

    elif y_pred[i]>1.5 and y_pred[i]<2.5 :

        y_pred[i]=2

    elif y_pred[i]>2.5 and y_pred[i]<3.5 :

        y_pred[i]=3

    else:

        y_pred[i]=4
from sklearn.metrics import confusion_matrix as cm

cm = cm(y_test,y_pred)

print(cm)
y_pred = model.predict_classes(X_test)
