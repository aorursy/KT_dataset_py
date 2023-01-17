from tensorflow import keras

model = keras.models.load_model('../input/indooroutdoor/classifier.h5')
import glob

testImages = glob.glob("../input/testingimages/*.jpg")
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.image as mpimg

from keras.preprocessing import image

%matplotlib inline
cols = 4

rows = np.ceil(len(testImages)/cols)

fig = plt.gcf()

fig.set_size_inches(cols * 4, rows * 4)

plt.figure(figsize = (cols * 4, rows * 4))

for i in range(len(testImages)):

    plt.subplot(rows, cols, i+1)

    img = image.load_img(testImages[i], target_size=(64, 64))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    eval_predictions = model.predict_classes(images)

    imagepath = mpimg.imread(testImages[i])

    plt.imshow(imagepath)

    plt.title("Indoor" if eval_predictions[0][0]==0 else "Outdoor")

    plt.axis('off')