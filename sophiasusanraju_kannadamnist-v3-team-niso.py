from __future__ import absolute_import, division, print_function, unicode_literals



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style("whitegrid")



import pandas as pd



from datetime import datetime



from sklearn import metrics

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix





from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



import keras

from keras.models import Sequential

from keras.layers import Dense, MaxPool2D, Conv2D, Flatten, Dropout, BatchNormalization

from keras.losses import categorical_crossentropy

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

validation = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

sample = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
train.head()
print('train:{}'.format(train.shape))

print('test:{}'.format(test.shape))

print('Dig_MNIST:{}'.format(validation.shape))



sns.countplot(train["label"])

plt.title("Distribution of Digit Samples in the Training Set")
validation.head()
sns.countplot(validation["label"])

plt.title("Distribution of Digit Samples in Validation Set")
test.head()
#Spliting off labels/Ids

Y_train = to_categorical(train.iloc[:,0])

X_train = train.iloc[:, 1:].values



X_valid = validation.iloc[:, 1:].values

y_valid = to_categorical(validation.iloc[:,0])



Y_test = to_categorical(test.iloc[:, 0])

X_test = test.iloc[:, 1:].values

#Normalizing the data

X_train = X_train/255

X_valid = X_valid/255



X_test = X_test/255
#Reshaping data

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)



X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)



X_test = X_test.reshape(test.shape[0], 28, 28, 1)
print(X_train.shape)

print(Y_train.shape)
#Visualizing The training data

fig, ax = plt.subplots(5, 10)

for i in range(5):

    for j in range(10):

        ax[i][j].imshow(X_train[np.random.randint(0, X_train.shape[0]), :, :, 0], cmap = plt.cm.binary)

        ax[i][j].axis("off")

plt.subplots_adjust(wspace=0, hspace=0)

fig.set_figwidth(15)

fig.set_figheight(7)

plt.show()
#Augmenting data

train_datagen = ImageDataGenerator(

    rotation_range=12,

    width_shift_range=0.25,

    height_shift_range=0.25,

    shear_range=12,

    zoom_range=0.25

)



valid_datagen = ImageDataGenerator(    

    rotation_range=12,

    width_shift_range=0.25,

    height_shift_range=0.25,

    shear_range=12,

    zoom_range=0.25)



valid_datagen_simple = ImageDataGenerator()
#X_train, x_valid_new, Y_train, y_valid_new = train_test_split(X_train, Y_train, test_size = 0.20, random_state=84)
#Build The Model

model = keras.Sequential([

    keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(32, kernel_size=5, activation='relu'),

    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.25),

    

  

    

    keras.layers.Conv2D(128, kernel_size=3, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(128, kernel_size=3, activation='relu'),

    keras.layers.BatchNormalization(),

   # keras.layers.MaxPooling2D(pool_size=(3, 3)),

    #keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.5),

    

    keras.layers.Conv2D(256, kernel_size=3, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(512, kernel_size=3, activation='relu'),

    keras.layers.BatchNormalization(),

    #keras.layers.MaxPooling2D(pool_size=(2, 2)),

   

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.5),

    

       

    keras.layers.Conv2D(1024, kernel_size = 3, strides=2, padding='same', activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

   

    keras.layers.Dense(1024,activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.2),

    

    keras.layers.Dense(10, activation='softmax')

])

# Take a look at the model summary

model.summary()
#Compile The Model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Train the Model

history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size = 1800),

                   epochs=70,validation_data = (valid_datagen.flow(X_valid, y_valid, batch_size = 30)),

                   verbose=1

                    )
#test_loss,

valid_loss,valid_acc = model.evaluate(X_valid,  y_valid, verbose=0)

print('Validation Accuracy: %.2f' %(valid_acc*100))

print('\n Validation Loss: %.2f' % (valid_loss))

train_loss, train_acc = model.evaluate(X_train,  Y_train, verbose = 1)

print('\nTraining accuracy: %.2f\n ' %(train_acc*100))
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.axis([0.0,70,0.60,1.00])

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.axis([0.0,70,0.0,2.0])

plt.show()
label_hot = model.predict(X_test)

label = np.argmax(label_hot,1)

id_ = np.arange(0,label.shape[0])
sample = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
sample.head()
submit = pd.DataFrame({'id':id_,'label':label})

print(submit.head(10))

submit.to_csv('submission.csv',index=False)