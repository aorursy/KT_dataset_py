
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os

import keras 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D
from keras import models
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
covid_dir='../input/deep-learning-competition-cs-2020/train/train/COVID19 AND PNEUMONIA/'
normal_dir='../input/deep-learning-competition-cs-2020/train/train/NORMAL'
covid_imgs=[]
labels=[]
img_width=128
img_height=128
for  dirname, _, imgs in os.walk(covid_dir):
    for img in imgs:
        abs_path = os.path.join(dirname, img)
        image=cv2.imread(abs_path)
        image=cv2.resize(image,(img_width,img_height))
        covid_imgs.append(image)
        labels.append(1)
for  dirname, _, imgs in os.walk(normal_dir):
    for img in imgs:
        abs_path = os.path.join(dirname, img)
        image=cv2.imread(abs_path)
        image=cv2.resize(image,(img_width,img_height))        
        covid_imgs.append(image)
        labels.append(0)
imgs=covid_imgs
test_dir='../input/deep-learning-competition-cs-2020/test/test'
test_imgs=[]
img_names=[]
for  dirname, _, imgs in os.walk(test_dir):
    for img in imgs:
        abs_path = os.path.join(dirname, img)
        img_names.append(img)
        image=cv2.imread(abs_path)
        image=cv2.resize(image,(img_width,img_height))
        test_imgs.append(image)
test_imgs=np.array(test_imgs)
print(len(covid_imgs))
X_train, X_test, y_train, y_test = train_test_split(covid_imgs, labels, test_size=0.3, random_state=42,stratify=labels)
x_train=np.array(X_train)
x_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)
input_img = Input(shape=(img_width, img_height, 3))
layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)
layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)
layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)
mid_1 = keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)
flat_1 = Flatten()(mid_1)
dense_1 = Dense(400, activation='relu')(flat_1)
dense_2 = Dense(200, activation='relu')(dense_1)
dense_3 = Dense(100, activation='relu')(dense_2)
output = Dense(2, activation='softmax')(dense_3)
model = Model([input_img], output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

ThisModel = model.fit(x_train, y_train, epochs=20,batch_size=64,verbose=1,validation_data=(x_test, y_test))
ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test)
print(ModelAccuracy)
y_pred = model.predict(test_imgs)
y_pred=[np.argmax(i) for i in y_pred]
data = list(zip(img_names, y_pred))
submission_df = pd.DataFrame(data, columns=['Image','Label'])
submission_df.head()
submission_df.to_csv("Submission.csv",index=False)