import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import os
model = tf.keras.models.load_model("../input/cat-or-dog-casual-model/casual_model.h5")
imgs = os.listdir("../input/dogs-vs-cats-redux-kernels-edition/test")
def get_img(path):

    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))

    arr = tf.keras.preprocessing.image.img_to_array(img)

    arr = np.divide(arr, 255.)

    return arr
for img in imgs[:10]:

    path = os.path.join("../input/dogs-vs-cats-redux-kernels-edition/test", img)

    arr = get_img(path)

    plt.imshow(arr)

    plt.show()

    pred = int(

        round(

            model.predict(arr.reshape(1, 224, 224, 3))[0][0]

        )

    )

    print(pred)