from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import os

import cv2

import pandas as pd

import numpy as np
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

 # Dropout(0.5),

  Dense(1, activation='sigmoid')

])



cnn_method1_nomedian.load_weights('../input/alzcnnpredictor/method1-nomedian-wholedataset.h5')
# read images from folder

directory = "/kaggle/input/alztest-images/method1-nomedian"

files = os.listdir(directory)

images = [file for file in files if os.path.splitext(file)[1] == '.jpg']

X_test = []

for image_name in images:

    grey = cv2.imread(os.path.join(directory,image_name),0)

    X_test.append(grey)
# preprocessing

# some images have different shape (384, 418) than others (384, 512) so, convert them by adding padding (47) to left and right

for i, img in enumerate(X_test):

    if img.shape == (384, 418):

        X_test[i]= cv2.copyMakeBorder(img.copy(),0,0,47,47,cv2.BORDER_CONSTANT,value=(0,0,0))

    if img.shape == (512, 384):

        X_test[i]= np.zeros((384, 512), np.uint8)





X_array = np.array(X_test)

#X_array = (X_array / 255)

X_array = np.expand_dims(X_array, axis=3)
# predictions

pred = cnn_method1_nomedian.predict_classes(X_array)

pred_df = pd.DataFrame(zip(images,pred.reshape(-1)), columns = ["filename","stalled"])



pred_df.to_csv("submission3.csv", encoding='utf-8', index=False)