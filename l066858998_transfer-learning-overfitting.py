import os

import json

import numpy as np

import urllib.request

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt
training_dataset_path = "../input/dogs-cats-images/dataset/training_set/"

validation_dataset_path = "../input/dogs-cats-images/dataset/test_set/"



training_dataset_cat_path = training_dataset_path + "cats"

training_dataset_dog_path = training_dataset_path + "dogs"



validation_dataset_cat_path = validation_dataset_path + "cats"

validation_dataset_dog_path = validation_dataset_path + "dogs"
print("Number of training images for dog: ", len(os.listdir(training_dataset_dog_path)))

print("Number of training images for cat: ", len(os.listdir(training_dataset_cat_path)))
print("Number of validation images for dog: ", len(os.listdir(validation_dataset_dog_path)))

print("Number of validation images for cat: ", len(os.listdir(validation_dataset_cat_path)))
training_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=90,

                                                                width_shift_range=0.2,

                                                                height_shift_range=0.2, 

                                                                shear_range=0.2,

                                                                zoom_range=0.2,

                                                                fill_mode="nearest",

                                                                horizontal_flip=True,

                                                                rescale=1/255)
training_generator = training_datagen.flow_from_directory(directory=training_dataset_path, 

                                                          target_size=(300, 300),

                                                          class_mode="categorical",

                                                          batch_size=32)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_generator = training_datagen.flow_from_directory(directory=validation_dataset_path, 

                                                          target_size=(300, 300),

                                                          class_mode="categorical",

                                                          batch_size=32)
# load pre-defined model architecture from keras

pre_trained_model = keras.applications.inception_v3.InceptionV3(include_top=False,

                                                    weights=None,

                                                    input_shape=(300, 300, 3))
# download pre-trained weight

pre_trained_weight_name = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

urllib.request.urlretrieve("https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", pre_trained_weight_name)
# load pre-trained weight to model

pre_trained_model.load_weights(pre_trained_weight_name)
# lock layer in pre-trained model

for layer in pre_trained_model.layers:

    layer.trainable = False
pre_trained_model.summary()
# the output of pre-trained model

pre_trained_model_output = pre_trained_model.output



# append customed layer to create new model

x = keras.layers.Flatten()(pre_trained_model_output)

x = keras.layers.Dense(units=1024, activation=tf.nn.relu)(x)



# create model without dropout layer

outputs = keras.layers.Dense(units=2, activation=tf.nn.softmax)(x)

model = keras.models.Model(inputs=pre_trained_model.input, outputs=outputs)



# create model with dropout layer

x = keras.layers.Dropout(0.3)(x)

outputs = keras.layers.Dense(units=2, activation=tf.nn.softmax)(x)

model_dropout = keras.models.Model(inputs=pre_trained_model.input, outputs=outputs)
model.summary()
model_dropout.summary()
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["acc"])
model_dropout.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["acc"])
pass
history = model.fit(x=training_generator,

                    epochs=100,

                    validation_data=validation_generator,

                    steps_per_epoch=32,

                    validation_steps=32)
with open("history.json", "w") as file:

    json.dump(history.history, file)
history_dropout = model_dropout.fit(x=training_generator,

                                    epochs=100,

                                    validation_data=validation_generator,

                                    steps_per_epoch=32,

                                    validation_steps=32)
with open("history_dropout.json", "w") as file:

    json.dump(history_dropout.history, file)
with open("history.json", 'r') as file:

    data = file.read()



history = json.loads(data)
with open("history_dropout.json", 'r') as file:

    data = file.read()



history_dropout = json.loads(data)
training_acc = history["acc"]

training_val_acc = history["val_acc"]

dropout_training_acc = history_dropout["acc"]

dropout_training_val_acc = history_dropout["val_acc"]

epochs = list(range(len(training_acc)))
plt.plot(epochs, training_acc, 'bo', label="Training Acc")

plt.plot(epochs, training_val_acc, 'b', label="Validation Acc")

plt.title("Training Acc vs Validation Acc without Dropout")

plt.legend()

plt.show()
plt.plot(epochs, dropout_training_acc, 'bo', label="Training Acc")

plt.plot(epochs, dropout_training_val_acc, 'b', label="Validation Acc")

plt.title("Training Acc vs Validation Acc with Dropout")

plt.legend()

plt.show()