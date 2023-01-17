# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

from keras.optimizers import SGD
%matplotlib inline

import matplotlib.pyplot as plt
#Loading csv into dataframe

train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
#Fetching labels

train_label = train['label']

test_label = test['label']



#Fetching pixel values

train_data =train.drop(columns=['label'])

test_data = test.drop(columns=['label'])
train_data.head()
# Reshape train and test data

train_data=train_data.values.reshape(-1,28,28,1)

test_data=test_data.values.reshape(-1,28,28,1)
# normalize the data

train_data = train_data/255.

test_data = test_data/255.
print(train_data.shape, test_data.shape)
# Analysis to see if the data is properly distributed for all the categories

train.label.sort_values().value_counts().plot(kind = 'bar')



len(train['label'].sort_values().unique()) # there 24 unique labels available
print(len(train_label), len(train_data))
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

train_label = label_binarizer.fit_transform(train_label)

test_label = label_binarizer.fit_transform(test_label)
print(train_label.shape, test_label.shape)
#divide data into train and validation

x_train, x_validation, y_train, y_validation = train_test_split(train_data, train_label,test_size=0.2, shuffle= True)
print(x_train.shape, x_validation.shape, y_train.shape, y_validation.shape)
# data augmentation

train_datagen = ImageDataGenerator(

                                rotation_range=40,

                                width_shift_range=0.2,

                                height_shift_range=0.2,

                                shear_range=0.2,

                                zoom_range=0.2,

                                horizontal_flip=True,

                                vertical_flip=True,

                                fill_mode='nearest')

validation_datagen = ImageDataGenerator()

train_datagen.fit(x_train)

validation_datagen.fit(x_validation)





training_gen = train_datagen.flow(x_train, y_train,batch_size=32)

validation_gen = validation_datagen.flow(x_validation, y_validation,batch_size=32)
#model



model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, (3,3),strides=(1,1), activation='relu', input_shape=(28,28,1)),

                                    tf.keras.layers.Conv2D(8, (3,3),strides=(1,1), activation='relu'),

                                    tf.keras.layers.BatchNormalization(),

                                    tf.keras.layers.ZeroPadding2D(padding=(1,1)),

                                    tf.keras.layers.Conv2D(16, (3,3),strides=(1,1), activation='relu'),

                                    tf.keras.layers.Conv2D(16, (3,3),strides=(1,1), activation='relu'),

                                    tf.keras.layers.BatchNormalization(),                                    

                                    tf.keras.layers.Conv2D(16, (3,3),strides=(1,1), activation='relu'),

                                    tf.keras.layers.Conv2D(16, (3,3),strides=(1,1), activation='relu'),

                                    tf.keras.layers.MaxPooling2D((2,2)),

                                    tf.keras.layers.Conv2D(16, (3,3),strides=(1,1), activation='relu'),

                                    tf.keras.layers.Conv2D(16, (3,3),strides=(1,1), activation='relu'),

                                    tf.keras.layers.MaxPooling2D((2,2)),                                    

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512, activation='relu'),

                                    tf.keras.layers.Dropout(0.2),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512, activation='relu'),

                                    tf.keras.layers.Dropout(0.2),                                    

                                    tf.keras.layers.Dense(24, activation='softmax')

                                   ])

#optimizer

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9)



# compile model

model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
model.summary()


# Callbacks



# Callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True), 

#            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)]



Callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)]



# Train the Model

history = model.fit_generator(

    training_gen,

    epochs=50,

    steps_per_epoch= len(x_train)/32,    

    validation_data=validation_gen,

    validation_steps = len(x_validation)/32,

    callbacks = Callbacks

)
# Plot the chart for accuracy and loss on both training and validation



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
model.evaluate(test_data, test_label)
prediction = model.predict(test_data)
prediction.shape
max_index_row = np.argmax(prediction, axis=1)
results = pd.Series(max_index_row,name="PredictedLabel")
submission = pd.concat([pd.Series(range(1,7173), name="image_id"),results], axis=1)
submission.to_csv("prediction.csv", index=False)
max_index_row.shape
test_label.shape
test['label']