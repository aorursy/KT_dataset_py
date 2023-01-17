# Unzipping files

from zipfile import ZipFile

with ZipFile('/kaggle/input/aerial-cactus-identification/test.zip', 'r') as zipObj:

   zipObj.extractall()

with ZipFile('/kaggle/input/aerial-cactus-identification/train.zip', 'r') as zipObj:

   zipObj.extractall()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Files saved into this location

print(os.listdir('/kaggle/working'))
import numpy as np

import pandas as pd

from keras.preprocessing import image

from keras import optimizers, models, layers

import matplotlib.pyplot as plt

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator



trainDir = '/kaggle/working/train'

testDir = '/kaggle/working/test'

trainCsvDir = '/kaggle/input/aerial-cactus-identification/train.csv'



trainDataFrame = pd.read_csv(trainCsvDir)

trainDataFrame.head()
datagen = ImageDataGenerator(rescale=1./255)

trainDataFrame.has_cactus = trainDataFrame.has_cactus.astype(str)

train_generator = datagen.flow_from_dataframe(dataframe=trainDataFrame[:15000], directory=trainDir, x_col='id',

                                             y_col='has_cactus', class_mode='binary', batch_size = 150, target_size=(32,32))

validation_generator = datagen.flow_from_dataframe(dataframe=trainDataFrame[15000:], directory=trainDir, x_col='id',

                                                  y_col='has_cactus', class_mode='binary', batch_size = 50, target_size=(32,32))
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3), padding='same'))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation='relu', padding='same'))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation='relu', padding='same'))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))



model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])


history=model.fit_generator(train_generator, steps_per_epoch=100, epochs=8, validation_data=validation_generator, validation_steps=50)
epochs = 8

acc = history.history['acc']

epochs_ = range(0,epochs)

plt.plot(epochs_, acc, label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')

acc_val =  history.history['val_acc']

plt.scatter(epochs_, acc_val, label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
acc = history.history['loss']

epochs_ = range(0,epochs)

plt.plot(epochs_, acc, label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')

acc_val = history.history['val_loss']

plt.scatter(epochs_, acc_val, label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()
submission = pd.DataFrame({'id':os.listdir(testDir)})



test_generator = datagen.flow_from_dataframe(dataframe=submission, directory=testDir, x_col='id',

                                                class_mode=None, batch_size=50, target_size=(32,32), shuffle=False)



predictions = model.predict_generator(test_generator)

submission['has_cactus'] = predictions

submission.to_csv('submission.csv', index=False)
import shutil

shutil.rmtree('../working/test')

shutil.rmtree('../working/train')

print(os.listdir('../working'))
