import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input, decode_predictions

%matplotlib inline



print(os.listdir("../input/"))
labels_train_df = pd.read_csv("../input/scenes-data/scene_labels_train.csv")







    

!cp ../input/keras-pretrained-models/* ~/.keras/models/

#fig, ax = plt.subplots(1, figsize=(12, 10))

#ax.imshow(img / 255.) 

#ax.axis('off')

#plt.show()
#img = image.load_img(path_train+'0c58107693263d32551209512d858246e925fe29.jpg', target_size=(224, 224))

#img = image.img_to_array(img)

#x = preprocess_input(np.expand_dims(img.copy(), axis=0))

#preds = resnet_model.predict(x)

#decode_predictions(preds, top=5)
def read_img(img_id, path, size):

    """Read and resize image.

    # Arguments

        img_id: string

        path: path of train or test data

        size: resize the original image.

    # Returns

        Image as numpy array.

    """

    img = image.load_img(join(path, img_id), target_size=size)

    img = image.img_to_array(img)

    return img
j = int(np.sqrt(NUM_CLASSES))

i = int(np.ceil(1. * NUM_CLASSES / j))

fig = plt.figure(1, figsize=(16, 16))

grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)

for i, (label_id,image_id) in enumerate(labels_train_df.values):

    ax = grid[i]

    img = read_img(image_id, path_train, (224, 224))

    ax.imshow(img / 255.)

    x = preprocess_input(np.expand_dims(img.copy(), axis=0))

    preds = resnet_model.predict(x)

    _, imagenet_class_name, prob = decode_predictions(preds, top=1)[0][0]

    ax.text(10, 180, 'ResNet50: %s (%.2f)' % (imagenet_class_name , prob), color='w', backgroundcolor='k', alpha=0.8)

    ax.text(10, 200, 'LABEL: %s' % label_id, color='k', backgroundcolor='w', alpha=0.8)

    ax.axis('off')

plt.show()