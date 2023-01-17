%matplotlib inline

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

import numpy as np
training_set_path = "/kaggle/input/digit-recognizer/train.csv"

test_set_path = "/kaggle/input/digit-recognizer/test.csv"

with open(training_set_path) as training_file:

    file = np.loadtxt(training_file, dtype=list, delimiter=',')

    #the first column of each row contains training label.

    training_labels = file[1:29401,0].astype('int32')

    #the columns from 1:785 contains value of 28x28 pixel image.

    training_images = file[1:29401,1:785].astype('int32')

    training_images = training_images.reshape(len(training_images),28,28)

    val_labels = file[29401:42001,0].astype('int32')

    val_images = file[29401:42001,1:785].astype('int32')

    val_images = val_images.reshape(len(val_images),28,28)



with open(test_set_path) as test_file:

    file = np.loadtxt(test_file,dtype=list,delimiter=',')

    testing_images = file[1:,:784].astype('int32')

    testing_images = testing_images.reshape(len(testing_images),28,28)



#normalising the training and validation data by dividing each pixel value with maximum pixel value.

#Here, each pixel value is gray scale, so value lies from 0 to 255.0

training_images = training_images/255.0

val_images = val_images/255.0



print(training_images.shape)

print(training_labels.shape)

print(val_images.shape)

print(val_labels.shape)

print(testing_images.shape)

training_images = np.expand_dims(training_images,axis=3)

val_images = np.expand_dims(val_images,axis=3)

testing_images = np.expand_dims(testing_images,axis=3)

print(training_images.shape)

print(val_images.shape)

print(testing_images.shape)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(28,(5,5),activation='relu',input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10,activation='softmax')

])

model.compile(optimizer='adam',

             loss = 'sparse_categorical_crossentropy',

             metrics=['acc'])

model.summary()



history = model.fit(training_images,training_labels,epochs=7,

                   validation_data=(val_images,val_labels))



loss, acc= model.evaluate(val_images,val_labels)

print('loss = ' + str(loss))

print('acc = ' + str(acc))
acc = history.history['acc']

val_acc = history.history['val_acc']

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
val_pred = model.predict_classes(val_images)

val_pred = val_pred.reshape(12600,1)

from sklearn.metrics import confusion_matrix

confusion_matrix(val_labels,val_pred,labels=[0,1,2,3,4,5,6,7,8,9])
test_predictions = model.predict_classes(testing_images)

test_predictions = test_predictions.reshape(28000,1)

idx = 237

img = testing_images[idx].reshape(28,28)

plt.imshow(img)

print(test_predictions[idx])
image_id = np.arange(1,28001).reshape(28000,1)

test_results = np.concatenate((image_id,test_predictions),axis=1)

np.savetxt('test_predictions.csv',test_results,fmt="%d",header="ImageId,Label",delimiter=",")