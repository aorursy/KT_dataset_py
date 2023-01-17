# Let's get the imports out of the way

import pandas as pd

import numpy as np

import cv2

np.random.seed(1234) 

from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt



df_train = pd.read_json('../input/train.json')



def get_scaled_imgs(df):

    imgs = []

    

    for i, row in df.iterrows():

        #make 75x75 image

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)

        

        # Rescale

        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())

        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())

        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())



        imgs.append(np.dstack((a, b, c)))



    return np.array(imgs)
def get_more_images(imgs):

    

    more_images = []

    vert_flip_imgs = []

    hori_flip_imgs = []

      

    for i in range(0,imgs.shape[0]):

        a=imgs[i,:,:,0]

        b=imgs[i,:,:,1]

        c=imgs[i,:,:,2]

        

        av=cv2.flip(a,1)

        ah=cv2.flip(a,0)

        bv=cv2.flip(b,1)

        bh=cv2.flip(b,0)

        cv=cv2.flip(c,1)

        ch=cv2.flip(c,0)

        

        vert_flip_imgs.append(np.dstack((av, bv, cv)))

        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

      

    v = np.array(vert_flip_imgs)

    h = np.array(hori_flip_imgs)

       

    more_images = np.concatenate((imgs,v,h))

    

    return more_images
def get_model():

    model=Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   

    return model
Xtrain = get_scaled_imgs(df_train)

Ytrain = np.array(df_train['is_iceberg'])



Xtr_more = get_more_images(Xtrain) 

Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))



model = get_model()

history_1 = model.fit(Xtr_more, Ytr_more, batch_size=32, epochs=10, verbose=1, validation_split=0.25)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(Xtrain, Ytrain, test_size=0.25)



X_train_more = get_more_images(X_train)

y_train_more = np.concatenate([y_train, y_train, y_train])

X_valid_more = get_more_images(X_valid)

y_valid_more = np.concatenate([y_valid, y_valid, y_valid])



model = get_model()

history_2 = model.fit(X_train_more, y_train_more, batch_size=32, epochs=10, verbose=1,

                     validation_data=(X_valid_more, y_valid_more))
plt.figure(figsize=(12,8))

plt.plot(history_1.history['val_loss'], label='bad validation')

plt.plot(history_2.history['val_loss'], label='good validation')

plt.title('Validation Loss by Epch')

plt.xlabel('Epoch')

plt.ylabel('Validation Loss')

plt.legend()

plt.show()