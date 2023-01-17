import cv2

import glob

import numpy as np

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_data_path = "../input/horses-or-humans-dataset/horse-or-human/train/"
horse_image_name = glob.glob(training_data_path + "horses/*png")

human_image_name = glob.glob(training_data_path + "humans/*png")
print("Number of horse images: ", len(horse_image_name))

print("Number of human images: ", len(human_image_name))
def show_image_label(images, label, is_path=True):

    

    idx = 0

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

    

    for i in range(5):

        for j in range(5):

            

            if is_path:

                img = cv2.imread(images[idx])

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            else:

                img = images[idx]

                

            idx += 1

            

            axes[i, j].set_title(label=label, color="green", fontsize=15)

            axes[i, j].set_xticks([])

            axes[i, j].set_yticks([])

            axes[i, j].imshow(img)
show_image_label(horse_image_name[0:25], "horse")
show_image_label(human_image_name[0:25], "human")
train_datagen = ImageDataGenerator(rotation_range=270,

                                   width_shift_range=0.1,

                                   # height_shift_range=0.5,

                                   zoom_range=0.15,

                                   # horizontal_flip=True,

                                   # vertical_flip=True,

                                   rescale=1/255)
training_generator = train_datagen.flow_from_directory(directory=training_data_path,

                                  target_size=(300, 300),

                                  batch_size=25,

                                  class_mode="categorical")
batch_data = training_generator.next()
training_generator.batch_index
show_image_label(batch_data[0], "Not Sure", is_path=False)
training_generator.reset()
training_generator.batch_index
inputs = keras.layers.Input(shape=(300, 300, 3))

x = keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, padding="same", activation=tf.nn.relu)(inputs)

x = keras.layers.MaxPooling2D(pool_size=2)(x)

x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same", activation=tf.nn.relu)(x)

x = keras.layers.MaxPooling2D(pool_size=2)(x)

x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation=tf.nn.relu)(x)

x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation=tf.nn.relu)(x)

x = keras.layers.MaxPooling2D(pool_size=2)(x)

x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation=tf.nn.relu)(x)

x = keras.layers.MaxPooling2D(pool_size=2)(x)

x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation=tf.nn.relu)(x)

x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation=tf.nn.relu)(x)

x = keras.layers.MaxPooling2D(pool_size=2)(x)

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(units=128, activation=tf.nn.relu)(x)

x = keras.layers.Dense(units=16, activation=tf.nn.relu)(x)

outputs = keras.layers.Dense(units=2, activation=tf.nn.softmax)(x)



model = keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["acc"])
class CustomeCallback(keras.callbacks.Callback):

    

    def on_epoch_end(self, epoch, logs):

        

        if logs["acc"] >= 0.97:

            print("Model's accuracy is enough !")

            self.model.stop_training = True
custom_callback = CustomeCallback()
model.fit(x=training_generator, epochs=50, verbose=1, callbacks=[custom_callback], steps_per_epoch=32)
label = ["horse", "human"]
def load_image(path):

    

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (300, 300))

    

    img = np.reshape(img, (1, 300, 300, 3))

    

    return img
img = load_image("../input/horse-breeds/01_005.png")

label[np.argmax(model.predict(img))]