def visualize_input(img, ax):

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y],2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y]<thresh else 'black')
from tensorflow import keras



(imagenes_entrenamiento, etiquetas_entrenamiento), (imagenes_test, etiquetas_test) = keras.datasets.mnist.load_data()

keras.datasets.mnist.load_data()
print(f'Número que representa la matriz matriz: {etiquetas_entrenamiento[1]}')

print(f'Cada elemento corresponde a un pixel: \n {imagenes_entrenamiento[1]}')
imagenes_entrenamiento[0].shape
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111)

ax.set_yticklabels([])

ax.set_xticklabels([])

plt.axis('off')

visualize_input(imagenes_entrenamiento[1], ax)
imagenes_entrenamiento = imagenes_entrenamiento/255.0

imagenes_test = imagenes_test/255.0
red_neuronal = keras.models.Sequential([

    keras.layers.Flatten(input_shape=[28, 28]),

    keras.layers.Dense(100, activation="sigmoid"),

    keras.layers.Dense(10, activation="softmax")

])
red_neuronal.compile(loss='sparse_categorical_crossentropy',

              optimizer=keras.optimizers.SGD(lr=0.1),

              metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping()

red_neuronal.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=10, validation_split=0.05, callbacks=[early_stopping])
red_neuronal.evaluate(imagenes_test, etiquetas_test)
red_neuronal_2 = keras.models.Sequential([

    keras.layers.Flatten(input_shape=[28, 28]),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(200, activation="selu", kernel_initializer="lecun_normal"),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(10, activation="softmax")

])
red_neuronal_2.compile(loss='sparse_categorical_crossentropy',

              optimizer=keras.optimizers.SGD(lr=0.05),

              metrics=['accuracy'])
imagenes_entrenamiento = imagenes_entrenamiento/255

red_neuronal_2.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=10, validation_split=0.1, callbacks=[early_stopping])
red_neuronal_2.evaluate(imagenes_test/255, etiquetas_test)
cnn = keras.models.Sequential([

        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape= (28,28, 1)),

        keras.layers.MaxPooling2D(),

        keras.layers.Flatten(),

        keras.layers.BatchNormalization(),

        keras.layers.Dense(200, activation="selu", kernel_initializer="lecun_normal"),

        keras.layers.BatchNormalization(),

        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),

        keras.layers.BatchNormalization(),

        keras.layers.Dense(10, activation="softmax")

])
cnn.compile(optimizer=keras.optimizers.SGD(lr=0.05), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
imagenes_test = imagenes_test.reshape(10000, 28, 28, 1)

imagenes_entrenamiento = imagenes_entrenamiento.reshape(60000, 28, 28, 1)

cnn.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=30, callbacks=[early_stopping])
cnn.evaluate(imagenes_test, etiquetas_test)
neural_network_on_steroids = keras.applications.resnet50.ResNet50(weights="imagenet")
neural_network_on_steroids.summary()
from urllib.request import urlopen, Request 

import matplotlib.pyplot as plt

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}

imagen_perro = plt.imread(urlopen(Request('https://cdn.pixabay.com/photo/2015/11/17/13/13/dogue-de-bordeaux-1047521_1280.jpg', headers=headers)), format='jpg')

castillo = plt.imread(urlopen(Request('https://www.audioguiaroma.com/imagenes/castillo-san-angelo.jpg', headers=headers)), format='jpg')
import tensorflow as tf 

import numpy as np



imagen_perro_crop = tf.image.resize_with_pad(imagen_perro, 224, 224, antialias=True)

castillo_crop = tf.image.resize_with_pad(castillo, 224, 224, antialias=True)

imagenes = keras.applications.resnet50.preprocess_input(np.array([imagen_perro_crop, castillo_crop]))
imagenes_test = imagenes_test.reshape(10000, 28, 28, 1);

imagenes_entrenamiento = imagenes_entrenamiento.reshape(60000, 28, 28, 1);
pred = neural_network_on_steroids.predict(imagenes)
top_K = keras.applications.resnet50.decode_predictions(pred, top=3)

for image_index in range(2):

    print("Image #{}".format(image_index))

    for class_id, name, y_proba in top_K[image_index]:

        print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))

    print()