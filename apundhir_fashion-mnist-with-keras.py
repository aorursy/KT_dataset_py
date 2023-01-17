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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")
print("Total Records in training data = %d" %(train.shape[0]) )

print("Total Features in training data = %d" %(train.shape[1]) )

print("Total Records in test data = %d" %(test.shape[0]) )

print("Total Features in test data = %d" %(test.shape[1]) )
train_data = train.drop('label', axis=1)

train_label = train['label']



test_data = test.drop('label', axis=1)

test_label = test['label']
train_data = train_data.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
g = sns.countplot(train_label)
import random



plt.figure(figsize=(10,5))

for i in range(10):  

    plt.subplot(1, 10, i+1)

    r = random.randint(0, train_data.shape[0])

    plt.imshow(train_data[r].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')
## Normalization

train_data = train_data / 255.0

test_data = test_data / 255.0
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        rotation_range=15,  ## Degree range for random rotations.

        zoom_range = 0.10,  ## Range for random zoom

        width_shift_range=0.1, ## fraction of total width

        height_shift_range=0.1, ## fraction of total height

        shear_range=0.1,

)



val_datagen = ImageDataGenerator()
## Bring target to categorical

from keras.utils.np_utils import to_categorical



labels = to_categorical(train_label, num_classes = 10)
from keras.models import Sequential

from keras.layers import Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Dense



model = Sequential()



model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())





model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
batch_size = 50

epochs = 50



train_generator = datagen.flow(

    train_data,

    labels,

    batch_size=batch_size

)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



model_name = 'model' + '/'

    

if not os.path.exists(model_name):

    os.mkdir(model_name)

        

filepath = model_name + 'model.h5'



checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)



LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)

callbacks_list = [checkpoint, LR]
## Lets keep 10% of the data for validation

from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(train_data, labels, test_size = 0.1)



model_hist = model.fit_generator(train_generator, steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, 

                    callbacks=callbacks_list,

                    validation_data = (X_val,Y_val), class_weight=None, workers=1, initial_epoch=0)
plt.plot(model_hist.history['loss'])

plt.plot(model_hist.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
test_data.shape
predicted_digits = model.predict(test_data).argmax(axis=-1)

result_df = pd.DataFrame()

result_df['ImageId'] = list(range(1,test_data.shape[0] + 1))

result_df['Label'] = predicted_digits

result_df.to_csv("submission.csv", index = False)
test_label = to_categorical(test_label, num_classes = 10)

score = model.evaluate(test_data, test_label, verbose=0)

y_pred_cnn = model.predict_classes(test_data)

print('Test loss:', score[0])

print('Test accuracy:', score[1])