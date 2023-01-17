# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd 

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/digit-recognizer/train.csv')

test_df = pd.read_csv('../input/digit-recognizer/test.csv')



print('The training dataset size : ',train_df.shape)

print('The test dataset size : ', test_df.shape)
train_df.head(5)
train_df.isnull().sum().describe()
test_df.isnull().sum().describe()
y_train = train_df['label']

#For x values we will drop label column as it is already been assigned 

X_train = train_df.drop(['label'], axis=1)

#we will delete train_df to free space though it can still be kept

del train_df



sns.countplot(y_train)

y_train.value_counts()
X_train.head(5)
X_train = X_train/255.0

test_df = test_df / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test_df = test_df.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state = 2)

print('The training set size after model selection : ',X_train.shape, y_train.shape)

print('The validation set size after model selection : ',X_validate.shape, y_validate.shape)
g = plt.imshow(X_train[0][:,:,0])
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs = 30 

batch_size = 64
#IMAGE AUGMENTATION

datagen = ImageDataGenerator(

        rotation_range=10, 

        zoom_range = 0.1,

        width_shift_range=0.1,  

        height_shift_range=0.1

        )



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_validate, y_validate),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
plt.figure(figsize=(17,17))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title('Train - Accuracy')
results = model.predict(X_validate)

results = np.argmax(results, axis=1)
results = pd.Series(results,name="Label")

results
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("CNN_Digit_recognizer.csv",index=False)

print('Successfull Submission!!')
submission