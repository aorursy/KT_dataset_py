import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage import data, io, filters
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import cv2

def CNN(num_character):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(224,224,3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same', activation="relu")) 
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_character, activation='softmax'))
    return model
bart = io.imread("../input/simpsons_dataset/simpsons_dataset/bart_simpson/pic_0016.jpg")
bart_resized = cv2.resize(bart,(224,224))
io.imshow(bart_resized)
directory = "../input/simpsons_dataset/simpsons_dataset/"
label = []
images = []
num_character = 5

for name in os.listdir(directory)[0:num_character]:
        character = os.listdir(directory+name)
        for c in character:
            images.append(directory+name+"/"+c)
            label.append(name)

le = preprocessing.LabelEncoder()
le.fit(np.array(label).reshape(-1,))
y_encoded = le.transform(np.array(label).reshape(-1,))

y = np.zeros((len(y_encoded),num_character))

for i, y_i in enumerate(y_encoded):
    y[i][y_i] = 1

X = np.zeros((len(y),224,224,3)).astype(np.float32)

for i,img in enumerate(images):
    image = io.imread(img)/255
    resized_image = cv2.resize(image,(224,224)).astype(np.float32)
    X[i]=resized_image
    
model = CNN(num_character)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
# train
for i in range (0,20):
    model.fit(X,y,epochs=5,verbose=1)
    model.save_weights('weights.h5')
bartest = directory+"bart_simpson/"+"pic_0016.jpg"
im = cv2.resize(io.imread(bartest),(224,224))
im = np.expand_dims(im, axis=0)
out = model.predict(im)
print(out)
print(le.inverse_transform(np.array([np.argmax(out)])))

