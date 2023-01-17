
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import random
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import img_to_array
from keras.layers import Flatten
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,MaxPool2D,MaxPooling2D,Dense
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras import regularizers

from keras import backend as K
import keras
from keras import optimizers
from sklearn.model_selection import train_test_split
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
data_root='/kaggle/input/covidct/'
path_positive_cases = os.path.join('/kaggle/input/covidct/CT_COVID/')
path_negative_cases = os.path.join('/kaggle/input/covidct/CT_NonCOVID/')
size = 128

#size = 64
def read_images(data):
    lst_images = []
    for i in range(len(data)):
        img = cv2.imread(data[i]) 
        img = cv2.resize(img, (size, size))     
        lst_images.append(img)
    return lst_images

data_positive = glob(os.path.join(path_positive_cases,"*.png"))

data_negative = glob(os.path.join(path_negative_cases,"*.png"))
data_negative.extend(glob(os.path.join(path_negative_cases,"*.jpg")))

imgs_positive  = read_images(data_positive)
imgs_negative  = read_images(data_negative)
labels_positive = [1] * len(imgs_positive)
labels_negative  = [0] * len(imgs_negative)
Y = labels_positive + labels_negative
X = imgs_positive + imgs_negative
print (len(Y),len(X))
# #randomly the data
# from sklearn.utils import shuffle

# X, Y = shuffle(X, Y)

# len(X),len(Y)
Y = np.asarray(Y)
X = np.asarray(X)
X = X.astype("float32")  
X = X / 255.0
X.shape
Y
X_trainn, X_test, y_trainn, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 256,stratify=Y)
y_test
X_train, X_valid, y_train, y_valid = train_test_split(X_trainn, y_trainn, test_size = 0.1, random_state = 256,stratify=y_trainn)
y_valid
X_model =keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(size,size,3), )

for layer in X_model.layers:
    layer.trainable =False 

model = Sequential()
# Block 1

# model.add(keras.layers.Conv2D(filters=64, kernel_size=3, kernel_regularizer=regularizers.l1(0.002), activation="relu", input_shape= (size,size,3)))
# model.add(keras.layers.MaxPool2D(pool_size=2))

# model.add(keras.layers.Conv2D(filters=128, kernel_size=3, kernel_regularizer=regularizers.l1(0.002), activation="relu"))
# model.add(keras.layers.MaxPool2D(pool_size=2))
# model.add(keras.layers.Conv2D(filters=256, kernel_size=3, kernel_regularizer=regularizers.l2(0.002), activation="relu"))
# model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(X_model)
model.add(keras.layers.AveragePooling2D(pool_size=(4, 4)))
model.add(keras.layers.Dropout(0.43))

model.add(keras.layers.Flatten())


model.add(keras.layers.Dense(64,activation="relu" ))
model.add(keras.layers.Dropout(0.43))

model.add(keras.layers.Dense(1, activation="sigmoid",kernel_initializer="glorot_uniform"))


trainAug = keras.preprocessing.image.ImageDataGenerator( rotation_range=15, fill_mode="nearest")


EPOCHS =140
learning_rate = 0.001
lr_decay = 1e-6
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=learning_rate)
# opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / EPOCHS)

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])




BS = 1
filepath="weights.best1.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=300)


callbacks_list = [checkpoint, early] #early
history = model.fit_generator( trainAug.flow(X_train, y_train, batch_size=BS), steps_per_epoch=len(X_train) // BS,
                        validation_data=(X_valid, y_valid), validation_steps=len(y_train) // BS,epochs=EPOCHS,
                      callbacks=callbacks_list )
# history = model.fit(X_train, y_train , epochs=20,batch_size =16 , verbose =2, validation_data=(X_valid, y_valid) )

testModel = model.evaluate(X_test, y_test)
print("Acuarcy = %.2f%%"%(testModel[1]*100))
print("Loss = %.2f%%"%(testModel[0]*100))
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.ylim(0,2)


plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict_classes(X_test)

classes = ["Class " + str(i) for i in range(2) if i != 9]
print(classification_report(y_test, predictions, target_names = classes))

cm = confusion_matrix(y_test,predictions)
cm
cm = pd.DataFrame(cm , index = [i for i in range(2) if i != 9] , columns = [i for i in range(2) if i != 9])
plt.figure(figsize = (5,5))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
