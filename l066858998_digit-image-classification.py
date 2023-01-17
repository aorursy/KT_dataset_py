import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt

import cv2
# download mnist dataset

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# shape of data

x_train.shape, y_train.shape
# image

x_train = x_train / 255

x_test = x_test / 255
# label

y_train = tf.one_hot(indices=y_train, depth=10)

y_test = tf.one_hot(indices=y_test, depth=10)
# this function can show images and its labels predicted by model

def show_img_label(images, true_labels, pred_labels):

    

    idx = 0

    

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

    for i in range(5):

        for j in range(5):

            

            true = int(tf.argmax(true_labels[idx]))

            pred = int(tf.argmax(pred_labels[idx]))

            

            if true == pred:

                color = "green"

            else:

                color = "red"

            

            axes[i, j].set_title(label="Label= "+str(true), fontsize=15, color=color)

            axes[i, j].set_xticks([])

            axes[i, j].set_yticks([])

            axes[i, j].imshow(images[idx])

            idx += 1

    

    
show_img_label(x_train, y_train, y_train)
inputs = keras.layers.Input(shape=(28, 28, 1))



hidden = keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, padding="same")(inputs)

hidden = keras.layers.BatchNormalization()(hidden)

hidden = keras.layers.Activation("relu")(hidden)

hidden = keras.layers.MaxPooling2D(pool_size=2)(hidden)



hidden = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same")(hidden)

hidden = keras.layers.BatchNormalization()(hidden)

hidden = keras.layers.Activation("relu")(hidden)

hidden = keras.layers.MaxPooling2D(pool_size=2)(hidden)



hidden = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same")(hidden)

hidden = keras.layers.BatchNormalization()(hidden)

hidden = keras.layers.Activation("relu")(hidden)

hidden = keras.layers.MaxPooling2D(pool_size=2)(hidden)



hidden = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(hidden)

hidden = keras.layers.BatchNormalization()(hidden)

hidden = keras.layers.Activation("relu")(hidden)

hidden = keras.layers.MaxPooling2D(pool_size=2)(hidden)



hidden = keras.layers.Flatten()(hidden)



hidden = keras.layers.Dropout(rate=0.2)(hidden)

outputs = keras.layers.Dense(units=10, activation="sigmoid")(hidden)



model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(0.0001), metrics=["accuracy"])



model.summary()
model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1)
show_img_label(x_test[40:65], y_test[40:65], model.predict(x_test[40:65]))
model.evaluate(x=x_test, y=y_test)