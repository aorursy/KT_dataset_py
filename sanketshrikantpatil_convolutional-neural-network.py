import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

import numpy as np
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',

               'dog', 'frog', 'horse', 'ship', 'truck']
IMG_INDEX = 3

plt.imshow(train_images[IMG_INDEX])

plt.xlabel(class_names[train_labels[IMG_INDEX][0]])

plt.show()
train_images, test_images = train_images / 255.0, test_images / 255.0
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))

model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(train_images, train_labels, validation_split = 0.1, epochs=6)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print(test_acc)
predictions = model.predict(test_images)
def show_prediction(img):

    i=img

    plt.figure()

    plt.imshow(test_images[i])

    plt.show()

    print("the prediction is",class_names[np.argmax(predictions[i])]+" and real picture is of ",class_names[np.int(test_labels[i])]) 
show_prediction(2222)