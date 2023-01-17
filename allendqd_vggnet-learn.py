import numpy as np

import tensorflow as tf

from tensorflow import keras

import os

from tensorflow.keras import backend as K

from tensorflow.keras import initializers

import h5py





def load_dataset():

	train_dataset = h5py.File('../input/workshop-week6/train_signs.h5', "r")

	train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features

	train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

	test_dataset = h5py.File('../input/workshop-week6/test_signs.h5', "r")

	test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features

	test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

	classes = np.array(test_dataset["list_classes"][:])  # the list of classes

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes







def vgg(input_shape=(64, 64, 3), classes=6):

	x_input = keras.layers.Input(input_shape)

	x = keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', activation='relu')(x_input)

	x = keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.MaxPool2D((2, 2), (2, 2))(x)



	x = keras.layers.Conv2D(128, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.Conv2D(128, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.MaxPool2D((2, 2), (2, 2))(x)



	x = keras.layers.Conv2D(256, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.Conv2D(256, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.Conv2D(256, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.MaxPool2D((2, 2), (2, 2))(x)



	x = keras.layers.Conv2D(512, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.Conv2D(512, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.Conv2D(512, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.MaxPool2D((2, 2), (2, 2))(x)



	x = keras.layers.Conv2D(512, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.Conv2D(512, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.Conv2D(512, (3, 3), (1, 1), padding='same', activation='relu')(x)

	x = keras.layers.MaxPool2D((2, 2), (2, 2))(x)



	x = keras.layers.Flatten()(x)



# 	x = keras.layers.Dense(4096, activation='relu')(x)

# 	x = keras.layers.Dropout(0.5)(x)

# 	x = keras.layers.Dense(4096, activation='relu')(x)

# 	x = keras.layers.Dropout(0.5)(x)

# 	x = keras.layers.Dense(1000, activation='relu')(x)

	x = keras.layers.Dense(6, activation='softmax')(x)



	model = keras.models.Model(inputs=x_input, outputs=x)

	return model





if __name__ == '__main__':

	train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

# 	train_set_x_orig = tf.image.resize_images(train_set_x_orig, (224, 224))

# 	test_set_x_orig = tf.image.resize_images(test_set_x_orig, (224, 224))

	train_set_x_orig = train_set_x_orig / 255.

	test_set_x_orig = test_set_x_orig / 255.



# 	with tf.Session() as sess:

# 		train_set_x_orig = train_set_x_orig.eval()

# 		test_set_x_orig = test_set_x_orig.eval()



	train_set_x_orig -= np.mean(train_set_x_orig, axis=0)

	test_set_x_orig -= np.mean(test_set_x_orig, axis=0)

	# train_set_x_orig = np.transpose(train_set_x_orig, axes=[0, 3, 1, 2])

	# test_set_x_orig = np.transpose(test_set_x_orig, axes=[0, 3, 1, 2])



	train_y = np.zeros((train_set_y_orig.shape[1], 6))

	for i in range(train_set_y_orig.shape[1]):

		train_y[i][train_set_y_orig[0][i]] = 1



	test_y = np.zeros((test_set_y_orig.shape[1], 6))

	for i in range(test_set_y_orig.shape[1]):

		test_y[i][test_set_y_orig[0][i]] = 1

	model = vgg()

	sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005)

	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(train_set_x_orig, train_y, epochs=20, batch_size=64)

	preds = model.evaluate(test_set_x_orig, test_y)

	model.save('vggnet.h5')

	print(preds)