import numpy as np

import pandas as pd

import os

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import cv2

import keras

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.models import Sequential

from keras import initializers, callbacks

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('dark')

%matplotlib inline


cell_images_path='../input/cell-images-for-detecting-malaria/cell_images'

parasitized_path=cell_images_path+'/Parasitized'

uninfected_path=cell_images_path+'/Uninfected'
parasitized_folder=os.listdir(parasitized_path)

uninfected_folder=os.listdir(uninfected_path)
X=[]

y=[]

dim=(128,128)

count=0

for image in parasitized_folder:

    try:

        image=cv2.imread(parasitized_path+os.sep+image, cv2.IMREAD_COLOR)

        image=cv2.resize(image, dim)

        X.append(image)

        y.append('Infected')

    except:

        continue

for image in uninfected_folder:

    try:

        image=cv2.imread(uninfected_path+os.sep+image, cv2.IMREAD_COLOR)

        image=cv2.resize(image, dim)

        X.append(image)

        y.append('Uninfected')

    except:

        continue
target=pd.Series(y, name='target')

print('Number of infected: {}'.format(target.value_counts()[0]))

print('Number of uninfected: {}'.format(target.value_counts()[1]))



figure=plt.figure(figsize=(8,6))

g1=sns.countplot(x=target)

g1.set_xticklabels(['Infected', 'Uninfected']);

g1.set_xlabel('');
X=np.array(X)

y=np.array(y)
encoder=LabelEncoder()

y=encoder.fit_transform(y)
X_train, X_test, y_train, y_test=train_test_split(X, y.astype(np.int8), test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
figure, ax=plt.subplots(5,10, figsize=(14,6))

plt.tight_layout()

for row in range(5):

    for col in range(10):

        num=np.random.randint(len(X_train))

        ax[row, col].imshow(X_train[num])

        ax[row, col].tick_params(labelleft=False,labelbottom=False)

        if y[num]==1:

            ax[row,col].set_title('uninfected')

        else:

            ax[row,col].set_title('infected')

            









    

    
figure, ax=plt.subplots(1,3,figsize=(14,6))

ax[0].imshow(X_train[0])

ax[0].tick_params(labelleft=False,labelbottom=False)

ax[0].set_title('original');



ax[1].imshow(cv2.rotate(X_train[0], cv2.ROTATE_90_CLOCKWISE))

ax[1].tick_params(labelleft=False,labelbottom=False)

ax[1].set_title('rotated 90 clockwise');



ax[2].imshow(cv2.rotate(X_train[0], cv2.ROTATE_90_COUNTERCLOCKWISE))

ax[2].tick_params(labelleft=False,labelbottom=False)

ax[2].set_title('rotated 270 clockwise');



def data_augmentation(X_train, y_train):

    X_train_augmentated=np.copy(X_train)

    y_train_augmentated=np.copy(y_train)

    for _ in range(0, len(X_train)):

        try: 

            X_train_augmentated=np.concatenate((X_train_augmentated, np.array([cv2.rotate(X_train[_], cv2.ROTATE_90_CLOCKWISE)]), np.array([cv2.rotate(X_train[_], cv2.ROTATE_90_COUNTERCLOCKWISE)])), axis=0)

            y_train_augmentated=np.concatenate((y_train_augmentated, np.array([y_train[_]]),np.array([y_train[_]])), axis=0)

        except MemoryError:

            return (X_train_augmentated, y_train_augmentated)    

    return (X_train_augmentated, y_train_augmentated)
'''import time

begin=time.time()

X_train_augmentated, y_train_augmentated=data_augmentation(X_train,y_train)

print(f'{int(time.time()-begin)}')'''

X_train_augmentated=X_train

y_train_augmentaed=y_train
X_train_augmentated.shape, y_train_augmentated.shape, X_train.shape, y_train.shape

model=Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(128,128,3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu' ))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(120, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(60, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(10, activation='relu'))

model.add(Dropout(0.1))



model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.01)

mc = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss',  save_best_only=True)
history=model.fit(X_train , y_train , epochs=4000, validation_split=0.2, callbacks=[es, mc])
print('best epoch: {}'.format(np.argmin(np.array(history.history['val_loss']))))

print('validation accuracy in best epoch: {:.4f}'.format(history.history['val_accuracy'][np.argmin(np.array(history.history['val_loss']))]))

print('validation loss in best epoch: {:.4f}'.format(np.array(history.history['val_loss']).min()))



figure, ax=plt.subplots(1,2, figsize=(14,6))

ax[0].plot(history.history['accuracy'])

ax[0].plot(history.history['val_accuracy'])

ax[0].set_xlabel('Epochs')

ax[0].set_ylabel('Accuracy')

ax[0].legend(['Train', 'Val'], loc='upper left')

ax[0].plot([np.argmin(np.array(history.history['val_loss']))], [history.history['val_accuracy'][np.argmin(np.array(history.history['val_loss']))]], marker='o', markersize=3, color="red")

ax[0].axhline(history.history['val_accuracy'][np.argmin(np.array(history.history['val_loss']))], ls='--',  linewidth=1,  color='red')

ax[0].axvline(np.argmin(np.array(history.history['val_loss'])), ls='--',  linewidth=1,  color='red')

ax[0].grid(True)



ax[1].plot(history.history['loss'])

ax[1].plot(history.history['val_loss'])

ax[1].set_xlabel('Epochs')

ax[1].set_ylabel('Loss')

ax[1].legend(['Train', 'Val'], loc='upper right')

ax[1].plot([np.argmin(np.array(history.history['val_loss']))], [np.array(history.history['val_loss']).min()], marker='o', markersize=3, color="red")

ax[1].axhline(np.array(history.history['val_loss']).min(), ls='--',  linewidth=1,  color='red')

ax[1].axvline(np.argmin(np.array(history.history['val_loss'])), ls='--',  linewidth=1,  color='red')

ax[1].grid(True)

from keras.models import load_model

best_model = load_model('best_model.h5')
test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print('Accuracy in test set: {:.4f}'.format(test_accuracy[1]))
predictions=best_model.predict_classes(X_test)
confusion_matrix(predictions, y_test)
print(classification_report(predictions, y_test))