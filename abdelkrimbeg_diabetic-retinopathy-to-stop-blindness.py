import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

train = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
# dataTest = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv')


# path_directorie = '/kaggle/input/aptos2019-blindness-detection/test_images/'
# size = 192

# images_test = []
# len_test = len(dataTest['id_code'])
# for i in tqdm(range(len_test)):
   
#     image = os.path.join(path_directorie+str(dataTest['id_code'][i])+'.png')
#     image = cv2.imread(image)
#     image = cv2.resize(image,(size,size))
#     images_test.append(image)
size = 192
path_directorie = '/kaggle/input/aptos2019-blindness-detection/train_images/'
images_train = []
lables = []
len_train = len(train['id_code'])
for i in tqdm(range(len_train)):
    image = os.path.join(path_directorie+str(train['id_code'][i])+'.png')
    image = cv2.imread(image)
    image = cv2.resize(image,(size,size))
    label = train['diagnosis'][i]
    images_train.append(image)
    lables.append(label)
X_train = np.array(images_train) 
# X_testt = np.array(images_test)
#lables = np.array(lables)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
lables=le.fit_transform(lables)

import keras
lables= keras.utils.to_categorical(lables,5)
lables
X_train = X_train /255
# X_testt = X_testt /255
X_train =X_train.reshape(-1,size,size,3)
# X_testt =X_testt.reshape(-1,size,size,3)

from sklearn.model_selection import train_test_split

X_train1 , X_valid, Y_train1,Y_valid = train_test_split(X_train , lables, test_size=0.25, random_state=7 )

X_train , X_test, Y_train,Y_test = train_test_split(X_train1 , Y_train1, test_size=0.25, random_state=7 )
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.regularizers import l2,l1
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,MaxPool2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D

from keras.layers import SeparableConv2D

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(Conv2D(input_shape=(size,size,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))



model.add(MaxPool2D(2,2))

model.add(Conv2D(32 , (3,3)  , activation = 'relu' ))
model.add(MaxPool2D(2,2 ))

model.add(Conv2D(64 , (3,3)  , activation = 'relu' ))
model.add(MaxPool2D((2,2) ))

model.add(Conv2D(128 , (1,1)  , activation = 'relu' ))
model.add(MaxPool2D((2,2) ))



model.add(Flatten())
model.add(Dense(units=64,activation="relu"))
model.add(keras.layers.Dropout(0.13))

model.add(Dense(5, activation='sigmoid'))

data_generator = keras.preprocessing.image.ImageDataGenerator( zoom_range=0.00005,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,
                                                             
                                                             
                                                             rotation_range=180,)
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.15,
# fill_mode="nearest"
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

EPOCHS = 90
BS =256
BATCH_SIZE =32
filepath="weights.best1.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# early = EarlyStopping(monitor="val_loss", mode="min", patience=300)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=5, min_lr=0.001)


es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, factor=0.5, min_lr=1e-6, verbose=1)

callbacks_list = [checkpoint, es] #early
history = model.fit_generator( data_generator.flow(X_train, Y_train , batch_size=BATCH_SIZE), steps_per_epoch=X_train.shape[0] / BATCH_SIZE,
     epochs=82,
     validation_data=(X_valid,Y_valid),
     callbacks=callbacks_list )
# history = model.fit(X_train, Y_train , epochs=180,batch_size =256 , verbose =1, callbacks=callbacks_list, validation_data=(X_valid, Y_valid) )
testModel = model.evaluate(X_test,Y_test)
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
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
predicted_classes = model.predict_classes(X_test)
rounded_labels=np.argmax(Y_test, axis=1)
confusionMatrix = confusion_matrix(rounded_labels, predicted_classes)
confusionMatrix = pd.DataFrame(confusionMatrix , index = [i for i in range(5) if i != 5] , columns = [i for i in range(5) if i != 5])
plt.figure(figsize = (8,8))
sns.heatmap(confusionMatrix,cmap= "OrRd_r", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
print(classification_report(rounded_labels, predicted_classes, target_names=target_names))
