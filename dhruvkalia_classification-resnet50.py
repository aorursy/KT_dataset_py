# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras as keras

model = keras.applications.resnet50.ResNet50(weights="imagenet")
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications import resnet50

from keras.applications.imagenet_utils import decode_predictions

path = "../input/starfish/asd.jpg"

# load an image in PIL format

original_image = load_img(path, target_size=(224, 224))

numpy_image = img_to_array(original_image)



# Convert the image into 4D Tensor (samples, height, width, channels) by adding an extra dimension to the axis 0.

input_image = np.expand_dims(numpy_image, axis=0)



# preprocess for resnet50

processed_image_resnet50 = resnet50.preprocess_input(input_image.copy())



# resnet50

predictions_resnet50 = model.predict(processed_image_resnet50)

label_resnet50 = decode_predictions(predictions_resnet50)

print (label_resnet50)