!pip install tensorflow-gpu==2.0.0-beta1
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
import matplotlib.pyplot as plt #for plotting

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

import tensorflow as tf
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
z_train = Counter(train['label'])

z_train
sns.countplot(train['label'])
#loading the dataset.......(Test)

test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
x_train = (train.ix[:,1:].values).astype('float32') # all pixel values

y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits



x_test = test.values.astype('float32')
# preview the images first

plt.figure(figsize=(12,10))

x, y = 10, 4

for i in range(40):  

    plt.subplot(y, x, i+1)

    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')

plt.show()
x_train = x_train/255.0

x_test = x_test/255.0
print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
X_train = x_train.reshape(x_train.shape[0], 28, 28,1)

X_test = x_test.reshape(x_test.shape[0], 28, 28,1)



print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
num_of_classes = 10



epochs = 20

batch_size = 32



input_shape = (28, 28, 1)
class NaiveModel(tf.keras.Model):

    def __init__(self):

        super(NaiveModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')

        self.flatten = tf.keras.layers.Flatten()

        self.d1 = tf.keras.layers.Dense(128, activation='relu')

        self.d2 = tf.keras.layers.Dense(num_of_classes, activation='softmax')

    

    def call(self, x):

        x = self.conv1(x)

        x = self.flatten(x)

        x = self.d1(x)

        

        return self.d2(x)
class DeepModel(tf.keras.Model):

    def __init__(self):

        super(DeepModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal')

        self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal')

        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.dropout1 = tf.keras.layers.Dropout(0.20)

        self.conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.conv4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.dropout2 = tf.keras.layers.Dropout(0.25)

        self.conv5 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.flatten = tf.keras.layers.Flatten()

        self.d1 = tf.keras.layers.Dense(128, activation='relu')

        self.bn = tf.keras.layers.BatchNormalization()

        self.d2 = tf.keras.layers.Dense(num_of_classes, activation='softmax')

    

    def call(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.maxpool(x)

        x = self.dropout1(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.maxpool(x)

        x = self.dropout2(x)

        x = self.conv5(x)

        x = self.dropout2(x)

        x = self.flatten(x)

        x = self.d1(x)

        x = self.bn(x)

        

        return self.d2(x)
# model = NaiveModel()

model = DeepModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy() ###



optimizer = tf.keras.optimizers.RMSprop() ###
train_loss = tf.keras.metrics.Mean(name='train_loss')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')



val_loss = tf.keras.metrics.Mean(name='val_loss')

val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
@tf.function

def train_step(images, labels):

    with tf.GradientTape() as tape:

        predictions = model(images)

        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



    train_loss(loss)

    train_accuracy(labels, predictions)
@tf.function

def val_step(images, labels):

    predictions = model(images)

    v_loss = loss_object(labels, predictions)



    val_loss(v_loss)

    val_accuracy(labels, predictions)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.1, # Randomly zoom image 

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=False,  # randomly flip images

    vertical_flip=False)  # randomly flip images
train_ds = tf.data.Dataset.from_tensor_slices(

    (X_train, Y_train)).shuffle(10000).batch(batch_size)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size)
# learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

#                                             patience=3, 

#                                             verbose=1, 

#                                             factor=0.5, 

#                                             min_lr=0.0001)
# model.compile(loss=loss_object, optimizer=optimizer)
# model.summary()
# datagen.fit()



for epoch in range(epochs):

    for images, labels in train_ds:

        train_step(images, labels)



    for val_images, val_labels in val_ds:

        val_step(val_images, val_labels)



    template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'

    print (template.format(epoch+1,

                           train_loss.result(),

                           train_accuracy.result()*100,

                           val_loss.result(),

                           val_accuracy.result()*100))
# h = model.fit_generator(x=X_train, y=Y_train, 

#                         batch_size=batch_size, 

#                         epochs=epochs, 

#                         validation_data=(X_val, Y_val), 

#                         verbose=1, 

#                         steps_per_epoch=X_train.shape[0] // batch_size, 

#                         callbacks=learning_rate_reduction)
# model.compile(optimizer, loss_object)



# final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

# print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
#get the predictions for the test data

Y_test = tf.argmax(model.call(X_test), axis=1)
ids = [x for x in range(1, Y_test.shape[0] + 1)]

pd_submit = pd.DataFrame({'ImageId':ids, 'Label':Y_test})

pd_submit.to_csv("submission.csv", index=False)