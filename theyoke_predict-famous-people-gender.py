import tensorflow as tf



import matplotlib.pyplot as plt

import seaborn

import PIL.Image as Image

import numpy as np

import os



seaborn.set()
# extract tar file to '/kaggle/lfw_funneled'

!tar -k -C /kaggle -xf /kaggle/input/lfwpeople/lfw-funneled.tgz
images = []

for dirpath, dirnames, filenames in os.walk('/kaggle/lfw_funneled'):

    if dirpath == '/kaggle/lfw_funneled':

        continue

    

    for filename in filenames:

        path = os.path.join(dirpath, filename)

        im = Image.open(path).resize((100, 100)).convert('L')

        images.append(np.asarray(im))



images = np.array(images)

images_scaled = images[..., np.newaxis] / 255.0  # expand for channel axis and scale
model = tf.keras.models.load_model('/kaggle/input/celebrity-faces-and-genders-imdb-dnn/model.h5')

probs = model.predict(images_scaled)

genders = np.argmax(probs, axis=1)
indices = np.arange(images.shape[0])

np.random.shuffle(indices)



plt.figure(figsize=(15,15))

for i in range(64):

    j = indices[i]

    confidence = (probs[j, genders[j]] - 0.5) * 2.0

    gender = 'Male' if genders[j] else 'Female'

    

    plt.subplot(8,8,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(images[j], cmap=plt.cm.binary_r)

    plt.xlabel(f'{confidence:.1%} {gender}')

plt.show()