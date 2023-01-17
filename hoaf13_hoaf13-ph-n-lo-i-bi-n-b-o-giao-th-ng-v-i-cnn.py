import cv2

import numpy as np

import keras

from keras.models import Sequential, load_model

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

from keras.optimizers import SGD

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

import os 

import glob

import pandas as pd 

from pandas import DataFrame as df 
def CNN_MODEL(input_shape, num_classes):

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding = 'Same', activation='relu'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding = 'Same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(units=64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model
X_train = []

y_train = []

y_labels = []

X_test = []

y_test = []

X_val = []

y_val = []

filename_test = os.listdir("../input/proptit-aif-homework-1/final_test/final_test/") 
BATCH_SIZE = 128 #Batch Gradient

NUM_CLASSES = 8   

EPOCHS = 15      

INPUT_SHAPE = (64,64,1) # Đồng bộ các ảnh về kích cỡ 64 x 64 và kênh màu Gray  
folder_name = [0,2,6,10,14,22,33,34]

for i in range(len(folder_name)):

    img_dir = "../input/proptit-aif-homework-1/final_train/final_train/" + str(folder_name[i]) 

    data_path = os.path.join(img_dir,'*g')

    files = glob.glob(data_path)

    tmp_length = 0

    for f in files:

        img = cv2.imread(f, 0)

        img = cv2.resize(img, (64,64)) # Đồng bộ kích cỡ ảnh

        img = cv2.equalizeHist(img)    # Điều chỉnh độ tương phản của ảnh

        img = np.array(img)

        X_train.append(img)

        tmp_length+=1

    y_train.extend([i]*tmp_length)

    y_labels.append(folder_name[i])



y_train = np_utils.to_categorical(y = y_train , num_classes=8)

y_val = np_utils.to_categorical(y = y_val, num_classes=8)

X_train = np.array(X_train).reshape(len(X_train),64,64,1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

img_dir = "../input/proptit-aif-homework-1/final_test/final_test"

test_path = os.path.join(img_dir,"*g")

files = glob.glob(test_path)



for f in files:

    img = cv2.imread(f,0)

    img = cv2.resize(img, (64,64))

    img = cv2.equalizeHist(img)

    img = np.array(img)

    X_test.append(img)



X_test = np.array(X_test)

X_test = X_test.reshape(X_test.shape[0],64,64,1)
model = CNN_MODEL(INPUT_SHAPE , 8)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
model.save("model.h5")

    
for i in range(len(X_test)):

    y_test.append(y_labels[np.argmax(model.predict(X_test[i].reshape(1,64,64,1)))])



output = pd.DataFrame({'class': y_test, 'path': filename_test})

output.to_csv('my_submission.csv', index=False)





print("Your submission was successfully saved!")   