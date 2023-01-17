# For CSV handling and numeric operations

import numpy as np

import pandas as pd



# For ML Model

import tensorflow as tf

import keras



# For splitting data for training and testing

from sklearn.model_selection import train_test_split



# For plotting graphs and images

import matplotlib.pyplot as plt



# For preprocessing images

from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv('../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data/A_Z Handwritten Data.csv')

dataset.rename(columns={'0':'labels'}, inplace=True)
labels = dataset['labels']

labels = labels.to_numpy()



images = dataset.drop('labels', axis=1)

images = images.to_numpy()
print('Labels data shape', labels.shape)

print('Images data shape', images.shape)
def alpha_mapper(int_arr):

    map_dict = {

    0:'A',1:'B',2:'C',3:'D',4:'E',

    5:'F',6:'G',7:'H',8:'I',9:'J',

    10:'K',11:'L',12:'M',13:'N',14:'O',

    15:'P',16:'Q',17:'R',18:'S',19:'T',

    20:'U',21:'V',22:'W',23:'X',24:'Y',

    25:'Z'}

    result = np.vectorize(map_dict.get)(int_arr)

    return result
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2)
selected = np.random.randint(x_test.shape[0], size=100)
fig, axes = plt.subplots(10, 10, figsize=(10, 10))

fig.subplots_adjust(wspace=0)

for a, image, true_label in zip(

        axes.flatten(), x_test[selected],

        alpha_mapper(y_test[selected])):

    a.imshow(image.reshape(28, 28), cmap='gray_r')

    a.text(0, 10, str(true_label), color="black", size=15)



    a.set_xticks(())

    a.set_yticks(())



plt.show()
preprocessor = MinMaxScaler()

preprocessor.fit(x_train)



x_train = preprocessor.transform(x_train)

x_test = preprocessor.transform(x_test)
fig, axes = plt.subplots(10, 10, figsize=(10, 10))

fig.subplots_adjust(wspace=0)

for a, image, true_label in zip(

        axes.flatten(), x_test[selected],

        alpha_mapper(y_test[selected])):

    a.imshow(image.reshape(28, 28), cmap='gray_r')

    a.text(0, 10, str(true_label), color="black", size=15)



    a.set_xticks(())

    a.set_yticks(())



plt.show()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')

x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')
print(x_train[0].shape)
model = keras.Sequential()



model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5), input_shape=(28, 28, 1), activation = 'relu'))

model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5), input_shape=(28, 28, 1), activation = 'relu'))

model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())



print('Shape after CNN :', model.output_shape)



model.add(keras.layers.Dense(1000, activation = 'relu'))

model.add(keras.layers.Dense(300, activation = 'sigmoid'))

model.add(keras.layers.Dense(100, activation = 'relu'))

model.add(keras.layers.Dense(26, activation = 'softmax'))



print('Final Output Shape :', model.output_shape)

assert model.output_shape[1] == 26
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fitting = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 100, epochs = 25, verbose = 2)
!mkdir -p saved_model

model.save('saved_model/handwriting_recognizer')

model.save('handwriting_recognizer.h5')
preds_trained = model.predict(x_test[selected])



fig, axes = plt.subplots(10, 10, figsize=(10, 10))

fig.subplots_adjust(wspace=0)

for a, image, true_label, pred_trained in zip(

        axes.flatten(), x_test[selected],

        alpha_mapper(y_test[selected]),

        alpha_mapper(np.argmax(preds_trained, axis=1))):

    a.imshow(image.reshape(28, 28), cmap='gray_r')

    a.text(0, 10, str(true_label), color="black", size=15)

    a.text(20, 26, str(pred_trained), color="blue", size=15)



    a.set_xticks(())

    a.set_yticks(())



plt.show()
print(preds_trained[0])

print(np.argmax(preds_trained[0]))
plt.plot(fitting.history['loss'])

plt.plot(fitting.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(fitting.history['accuracy'])

plt.plot(fitting.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()