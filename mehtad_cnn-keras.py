# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2



X_img = []

y_p = []

def create_training_set(label, path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (150,150))

    X_img.append(np.array(img))

    y_p.append(str(label))
df_train = pd.read_csv('../input/train.csv')
X = df_train['id_code']

y = df_train['diagnosis']
from tqdm import tqdm



TRAIN_DIR = '../input/train_images'

for id_code, diagnosis in tqdm(zip(X,y)):

    path = os.path.join(TRAIN_DIR, '{}.png'.format(id_code))

    create_training_set(diagnosis, path)
from keras.utils import to_categorical



Y = to_categorical(y_p)

X= np.array(X_img)

X=X/255
from keras.preprocessing.image import ImageDataGenerator



feat_extraction = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)



# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

feat_extraction.fit(X)
from sklearn.model_selection import train_test_split



X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=22)



    
y.hist()
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense, GaussianDropout

from keras.constraints import maxnorm

from keras import regularizers, optimizers



model = Sequential()



model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(150,150,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(GlobalAveragePooling2D())



model.add(Dense(5, activation='softmax'))

model.summary()
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop



model.compile(optimizer= Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
batch_size =50

epochs= 50
from keras.callbacks import ModelCheckpoint



checkpointer =  ModelCheckpoint(filepath= 'CNN_keras.hdf5', verbose=1, save_best_only=True)



#model_history = model.fit_generator(feat_extraction.flow(X_train, Y_train, batch_size=batch_size),

 #        epochs= epochs, validation_data=feat_extraction.flow(X_valid, Y_valid, batch_size= batch_size),

  #       callbacks= [checkpointer], verbose=1, steps_per_epoch=X_train.shape[0]//batch_size, validation_steps=X_train.shape[0]//batch_size )





model_history=model.fit(X_train, Y_train, 

          validation_data=(X_valid, Y_valid),

          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)





import matplotlib.pyplot as plt

# list all data in history

print(model_history.history.keys())

# summarize history for accuracy

plt.plot(model_history.history['acc'])

plt.plot(model_history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
os.listdir('../input/test_images/')[0:5]
test_image = cv2.imread('../input/test_images/3d4d693f7983.png', cv2.IMREAD_COLOR)

test_image = cv2.resize(test_image, (150,150))

import matplotlib.pyplot as plt



plt.imshow(test_image)
test_X = np.array(test_image)

test_X = test_X/255
pred_test= model.predict(np.expand_dims(test_X,axis=0))
pred_test
test_df = pd.read_csv('../input/test.csv')

test_df.head()
test_ids = test_df['id_code']
test_images = []

def create_test_set(path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img = cv2.resize(img, (150,150))



    test_images.append(np.array(img))
from tqdm import tqdm

for id_code in tqdm(test_ids):

    path = os.path.join('../input/test_images','{}.png'.format(id_code))

    create_test_set(path)
from keras.models import load_model

model=load_model('CNN_keras.hdf5')
test_X=np.array(test_images)

test_X=test_X/255

feat_extraction.fit(test_X)

predictions=model.predict(test_X)
pred = np.argmax(predictions, axis=1)

pred
np.unique(pred)
submission_cnn = pd.DataFrame({'id_code' : test_ids , 'diagnosis' : pred})
submission_cnn.head()
submission_cnn.to_csv("submission.csv",index=False)