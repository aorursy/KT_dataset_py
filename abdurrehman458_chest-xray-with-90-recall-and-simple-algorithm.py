# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras import backend as K
from tensorflow.keras.optimizers import SGD , RMSprop
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
#         print(os.path.join(dirname, filename))
print(tf.__version__)
# Any results you write to the current directory are saved as output.
# Augmentation

base_dir=os.path.join("../input/chest-xray-pneumonia/chest_xray/chest_xray/")
train_dir=os.path.join(base_dir,"train")
val_dir=os.path.join(base_dir,"val")
print(train_dir, val_dir, sep='\n')

IMG_SHAPE=150
batch_size=64

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=15,
                    horizontal_flip=True
                    )


train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE),
                                                )

image_gen_val = ImageDataGenerator(rescale=1./255,
                   rotation_range=15,
                   horizontal_flip=True)

val_data_gen = image_gen_val.flow_from_directory(batch_size=16,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 )
print(train_data_gen.class_indices)
# model copied from https://github.com/deadskull7/Pneumonia-Diagnosis-using-XRays-96-percent-Recall

def swish_activation(x):
    return (K.sigmoid(x) * x)

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(150,150,3)))
model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(150,150,3)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation=swish_activation))
model.add(Dropout(0.4))
model.add(Dense(2 , activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.00005),
                  metrics=['accuracy'])

print(model.summary())
epochs=6
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(16)))
)
# from keras.models import load_model
model.save('/kaggle/working/m.h5')
# history=load_model('/kaggle/working/m.h5')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# generating test data set with labels

from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
test_list_pne=os.listdir("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/")
test_dir_pne=os.path.join("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/")
test_list_nor=os.listdir("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/")
test_dir_nor=os.path.join("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/")
test_x=[]
test_y=[]
for name in tqdm(test_list_pne):
    # predicting images
    path = test_dir_pne + name
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    test_x.append(x)
    test_y.append(1)
    
for name in tqdm(test_list_nor):
    # predicting images
    path = test_dir_nor + name
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    test_x.append(x)
    test_y.append(0)

test_x=np.array(test_x)
test_y=np.array(test_y)
test_y= to_categorical(test_y, 2)

print("Total number of test examples: ", test_x.shape)
print("Total number of labels:", test_y.shape)
# print(test_y)
test_loss, test_score = model.evaluate(test_x, test_y)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)

# prediction on test data

preds = model.predict(test_x)
preds = np.argmax(preds, axis=-1)
test_y= np.argmax(test_y,axis=-1) 
print("test_y= ", test_y.shape)
print("Pred= ", preds.shape)
# print(preds)
# pred=[]
# for p in preds:
#     if p == 1:
#         pred.append(1)
#     else:
#         pred.append(0)
# pred=np.array(pred)
# print("pred shape= ", pred.shape)
# # print(pred)
        


# CM = confusion_matrix(test_y, pred)
# fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))
# plt.show()

cm  = confusion_matrix(test_y, preds)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
# Ignore these cells they are just for testing

test_list=os.listdir("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/")
test_dir=os.path.join("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/")
# print(test_dir)
from keras.preprocessing import image
import skimage
from skimage.transform import resize
import cv2 
from tqdm import tqdm_notebook as tqdm
X=[]
T=0
F=0
immg=[]

for name in tqdm(test_list):
    # predicting images
    path = test_dir + name
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
    immg.append(x)
#     classes = model.predict(images)
#     if classes[0]>0.5:
#         F=F+1
#     else:
#         T=T+1

immge=np.array(immg)        
classes = model.predict_classes(np.squeeze(immge))
print(np.squeeze(immge).shape)
print(classes.shape)
for i in classes:
    if i>0.5:
        F=F+1
    else:
        T=T+1
print("T={} , F={}".format(T,F) )
T=0
F=0
# Ignore these cells they are just for testing

for name in tqdm(test_list):
    # predicting images
    path = test_dir + name
    img = cv2.imread(path)
    if img is not None:
        img = skimage.transform.resize(img, (150, 150, 3))
        #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
        img = np.asarray(img)
        X.append(img)
X = np.asarray(X)
print(X.shape)
classes = model.predict(X)
print(classes.shape)
for i in classes:
    if i>0.5:
        F=F+1
    else:
        T=T+1
print("T={} , F={}".format(T,F) )
T=0
F=0
    