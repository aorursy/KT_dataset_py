import os

from tqdm.notebook import tqdm_notebook as tqdm

import cv2

import numpy as np
import keras
CLASS = {

    'Black-grass': 0,

    'Charlock': 1,

    'Cleavers': 2,

    'Common Chickweed': 3,

    'Common wheat': 4,

    'Fat Hen': 5,

    'Loose Silky-bent': 6,

    'Maize': 7,

    'Scentless Mayweed': 8,

    'Shepherds Purse': 9,

    'Small-flowered Cranesbill': 10,

    'Sugar beet': 11

}



INV_CLASS = {CLASS[j]:j for j in CLASS}
def preprop_img(image_path, verbose=0):

    if verbose:

        print(image_path)

    img=cv2.imread(image_path)

    img=cv2.resize(img, (128,128))

    return img
X=[]

Y=[]

BASE='../input/plant-seedlings-classification/train'

for i in tqdm(os.listdir(BASE), total=len(CLASS)):

    for j in os.listdir(os.path.join(BASE,i)):

        X.append(preprop_img(os.path.join(BASE,i,j)))

        Y.append(CLASS[i])

X=np.array(X)

Y=np.array(Y)

print(X.shape, Y.shape)
# one-hot encode

Y_oh=keras.utils.to_categorical(Y,len(CLASS))

print(Y_oh.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y_oh, shuffle=True, test_size=0.2)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
model = keras.applications.ResNet50(input_shape=(128,128, 3), classes=12, weights=None)



# For a multi-class classification problem

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, validation_data=(X_test, Y_test), shuffle=True, epochs=25)
model.save_weights('Resnet50_plant_seedling.h5')