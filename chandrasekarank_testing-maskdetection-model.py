# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow import keras

nmodel = keras.models.load_model('../input/maskdetectionmodel/home/mask')
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from tensorflow.python.keras.applications.resnet import preprocess_input

import cv2

import matplotlib.pyplot as plt 

%matplotlib inline
def img_path(path):

    for dirname, _, filenames in os.walk(path):

        l = []

        y = []

        for filename in filenames:

            l.append(os.path.join(dirname, filename))

            y.append(os.path.join(dirname, filename).split("/")[-2])

        return (l, y)



image_size = 150

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)



def disp_img_with_path(test_img_path, title = "my picture"):

    image = cv2.imread(test_img_path)

    image = image[:, :, [2, 1, 0]]

    plt.imshow(image)

    plt.title(title)

    plt.show()

    (h, w, d) = image.shape

    print("width={}, height={}, depth={}".format(w, h, d))
img_paths = ['../input/prajna-bhandary-face-mask-detection-dataset/data/train/with_mask/1-with-mask.jpg', '../input/prajna-bhandary-face-mask-detection-dataset/data/train/without_mask/10.jpg', '../input/prajna-bhandary-face-mask-detection-dataset/data/train/without_mask/108.jpg'] 

test_img = read_and_prep_images(img_paths)

preds = nmodel.predict(test_img)

print(preds)
for img_path in img_paths:

    disp_img_with_path(test_img_path = img_path, title = "test_img")
labels_dict={1:'without_mask',0:'with_mask'}

for result in preds:

    label=np.argmax(result)

    print(labels_dict[label])