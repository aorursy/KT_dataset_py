# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
flowers=pd.read_csv('../input/flower-color-images/flower_images/flower_images/flower_labels.csv')

flowers_images=flowers['file']

flowers_label=flowers['label'].values
print(flowers_images.shape[0])

#flowers_images=flowers_images.reshape(210,3)

print(flowers_label.shape)
import cv2

from matplotlib import pyplot as plt

x=np.random.randint(210)

image=cv2.imread('../input/flower-color-images/flower_images/flower_images/'+flowers_images[x])

plt.figure(figsize=(3,3))

plt.imshow(image)

print(flowers_label[x])

import keras

from tqdm import tqdm

from keras.preprocessing import image as keras_image

def path_to_tensor(img_path):

    img=keras_image.load_img('../input/flower-color-images/flower_images/flower_images/'+img_path,target_size=(28,28))

    x=keras_image.img_to_array(img)

    return np.expand_dims(x,axis=0)



def paths_to_tensors(img_paths):

    list_of_tensors=[path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)
flowers_tensors=paths_to_tensors(flowers_images)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(flowers_tensors,flowers_label,test_size=0.2,random_state=1)
[x_train.shape,y_train.shape,x_test.shape,y_test.shape]
img_rows=28

img_cols=28

input_shape=(img_rows,img_cols,3)

x_train=x_train.astype('float32')

x_test=x_test.astype('float32')

x_train/=255

x_test/=255
y_train=keras.utils.to_categorical(y_train)

y_test=keras.utils.to_categorical(y_test)
y_train.shape


from keras.layers import Dense, Dropout, Flatten

from keras.layers import MaxPooling2D, Conv2D

from keras.layers import BatchNormalization

from keras.models import Model, Sequential, load_model

from keras.optimizers import SGD, RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
num_classes=10

model=Sequential()



model.add(Conv2D(64,kernel_size=(3,3),padding='same',input_shape=input_shape,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(num_classes,activation='softmax'))

print(model.summary())
from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping

earlystopping=EarlyStopping(monitor='val_loss',min_delta=0,patience=3,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy'])
callbacks=[earlystopping]

history=model.fit(x_train,y_train,epochs=25,batch_size=32,validation_data=(x_test,y_test))
score=model.evaluate(x_test,y_test,verbose=0)

print("LOSS : ", score[0])

print("ACCURACY : ", score[1])
data_generator = keras_image.ImageDataGenerator(shear_range=0.2, 

                                                zoom_range=0.3,

                                                rotation_range=30,

                                                width_shift_range=20,

                                                height_shift_range=20,

                                                horizontal_flip=True)

cnn_history =model.fit_generator(data_generator.flow(x_train, y_train, batch_size=32),

                                               steps_per_epoch=189, epochs=10, callbacks=callbacks,

                                               validation_data=(x_test, y_test))
score=model.evaluate(x_test,y_test,verbose=0)

print("LOSS : ", score[0])

print("ACCURACY : ", score[1])
Y_pred = model.predict(x_test)

y_pred = np.argmax(Y_pred, axis=1)

y_pred
submission=pd.DataFrame({'Label' : y_pred})

submission.head()
filename = 'Flower Color Classifier.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)