import numpy as np

import pandas as pd

import time

import cv2

import os

from sklearn.metrics import matthews_corrcoef

from sklearn.model_selection import train_test_split



directory = "/kaggle/input/micro-images/method1-nomedian"



# read images from folder

files = os.listdir(directory)

images = [file for file in files if os.path.splitext(file)[1] == '.jpg']

X = []

for image_name in images:

    grey = cv2.imread(os.path.join(directory,image_name),0)

    X.append(grey)

    

# read labels

labels_df = pd.read_csv("/kaggle/input/alz-metadata/train_labels.csv", sep=',')

y = [labels_df.loc[labels_df['filename']==os.path.splitext(image)[0]+".mp4", 'stalled'].squeeze() for image in images]



# some images have different shape (384, 418) than others (384, 512) so, convert them by adding padding (47) to left and right

for i, img in enumerate(X):

    if img.shape == (384, 418):

        X[i]= cv2.copyMakeBorder(img.copy(),0,0,47,47,cv2.BORDER_CONSTANT,value=(0,0,0))

    if img.shape == (512, 384):

        X[i]= np.zeros((384, 512), np.uint8)

        

# convert datasets to np arrays

X_array = np.array(X)

y_array = np.array(y)



# Normalize the images.

#X_array = (X_array / 255)



# train-test split

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size = 0.2, stratify = y, random_state = 42)



#reshape data to fit model

X_train = np.expand_dims(X_train, axis=3)

X_test = np.expand_dims(X_test, axis=3)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import keras.backend as K



#create model

model = Sequential()



#add model layers

model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(384,512,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=3, activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=3, activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

#model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))





# custom matthew's correlation function

def matthews_correlation(y_true, y_pred):

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tp = K.sum(y_pos * y_pred_pos)

    tn = K.sum(y_neg * y_pred_neg)



    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg)



    numerator = (tp * tn - fp * fn)

    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



    return 1.0 - numerator / (denominator + K.epsilon())



#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC', matthews_correlation])



#train and test the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)



#train and test the model on whole dataset

#model.fit( np.expand_dims(X_array, axis=3), y_array, epochs=3)
from sklearn.metrics import matthews_corrcoef



# matthew's correlation score on test set

pred = model.predict_classes(X_test)

pred_score = matthews_corrcoef(y_test, pred.reshape(-1))

print(pred_score)
# save model

model.save_weights('method1-nomedian-wholedataset.h5')



# Load the model's saved weights.

cnn_method1_nomedian = Sequential([

  Conv2D(32, kernel_size=3, activation="relu", input_shape=(384,512,1)),

  MaxPooling2D(pool_size=(2, 2)),

  Conv2D(32, kernel_size=3, activation="relu"),

  MaxPooling2D(pool_size=(2, 2)),

  Conv2D(64, kernel_size=3, activation="relu"),

  MaxPooling2D(pool_size=(2, 2)),

  Flatten(),

  Dense(64, activation='relu'),

  #Dropout(0.5),

  Dense(1, activation='sigmoid')

])





cnn_method1_nomedian.load_weights('method1-nomedian-wholedataset.h5')
# predict first 4 images in the test set

cnn_method1_nomedian.predict(X_test[:4])
# actual results for first 4 images in test set

y_test[:4]
import tensorflow as tf



tf.keras.backend.clear_session()