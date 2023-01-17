import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt #Ploting charts
from glob import glob #retriving an array of files in directories
from keras.models import Sequential #for neural network models
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical #For One-hot Encoding
from keras.optimizers import Adam, SGD, RMSprop #For Optimizing the Neural Network
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
#Cheking datasets
import os
paths = os.listdir(path="../input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL")
print(paths)
path_train = "../input/chestxray-images/xray/Xray/train"
path_val = "../input/chestxray-images/xray/Xray/val"
path_test = "../input/chestxray-images/xray/Xray/test"
path_trainD2 = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train"
path_valD2 = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val"
path_testD2 = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test"
img = glob(path_trainD2+"/PNEUMONIA/*.jpeg") #Getting all images in this folder
img = np.asarray(plt.imread(img[0]))
plt.imshow(img)
img = glob(path_train+"/NORMAL/*.jpeg") #Getting all images in this folder
img = np.asarray(plt.imread(img[0]))
plt.imshow(img)
img.shape
#Data preprocessing and analysis
classes = ["NORMAL", "PNEUMONIA"]
train_data = glob(path_trainD2+"/NORMAL/*.jpeg")
train_data += glob(path_trainD2+"/PNEUMONIA/*.jpeg")
data_gen = ImageDataGenerator() #Augmentation happens here
train_batches = data_gen.flow_from_directory(path_trainD2, target_size = (224, 224), classes = classes, class_mode = "categorical")
val_batches = data_gen.flow_from_directory(path_valD2, target_size = (224, 224), classes = classes, class_mode = "categorical")
test_batches = data_gen.flow_from_directory(path_testD2, target_size = (224, 224), classes = classes, class_mode = "categorical")
train_batches.image_shape

K.clear_session
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=train_batches.image_shape))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2))) # 112X112 pixels na de pooling

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2))) #56X56 pixels na de pooling

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2))) #28X28 pixels na de pooling

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2))) #14X14 pixels na de pooling

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2, activation='softmax'))
#Viewing the summary of the model
model.summary()
optimizer = Adam(lr = 0.001)
early_stopping_monitor = EarlyStopping(patience = 2, monitor = "val_acc", mode="max", verbose = 1)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = model.fit_generator(epochs=5, callbacks=[early_stopping_monitor], shuffle=True, validation_data=val_batches, generator=train_batches, steps_per_epoch=500, validation_steps=10,verbose=1)
prediction = model.predict_generator(generator=train_batches, verbose=1, steps=100)
prediction = model.predict_generator(generator=test_batches, verbose=1, steps=100)
Y_pred = model.predict_generator(test_batches,test_batches.samples  // 32+1)
y_pred = np.argmax(Y_pred, axis=1)
prediction
print('Confusion Matrix')
#cm = confusion_matrix(test_batches.classes, y_pred)
cm  = confusion_matrix(test_batches.classes, y_pred)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()
print('Classification Report')
target_names = ['Normal','Pneumonia']
print(classification_report(test_batches.classes, y_pred, target_names=target_names))
'''
Source: Jason Brownlee
Site: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
'''

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
# summarize history for loss
'''
Source: Jason Brownlee
Site: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()