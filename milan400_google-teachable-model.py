# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import tensorflow.keras

from PIL import Image, ImageOps

import numpy as np

import cv2

import matplotlib.pyplot as plt
# Disable scientific notation for clarity

np.set_printoptions(suppress=True)
# Load the model

model = tensorflow.keras.models.load_model('/kaggle/input/keras-model-google/keras_model.h5')
def show_image_label(path):

    # Create the array of the right shape to feed into the keras model

    # The 'length' or number of images you can put into the array is

    # determined by the first position in the shape tuple, in this case 1.

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    

    

    # Replace this with the path to your image

    image = Image.open(path)



    #resize the image to a 224x224 with the same strategy as in TM2:

    #resizing the image to be at least 224x224 and then cropping from the center

    size = (224, 224)

    image = ImageOps.fit(image, size,Image.ANTIALIAS)



    #turn the image into a numpy array

    image_array = np.asarray(image)



    # Normalize the image

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1



    # Load the image into the array

    data[0] = normalized_image_array



    emotion_dict = {0:'Anger',1:'contempt',2:'disgust',3:'fear',4:'happy',5:'sadness',6:'surprise'}



    imgplot = plt.imshow(image)

    plt.show()



    # run the inference

    prediction = np.argmax(model.predict(data),axis=1)

    print(emotion_dict[prediction[0]])
test_imge_list = []

for dirname, _, filenames in os.walk('/kaggle/input/test-images/test_image'):

    for filename in filenames:

        a = os.path.join(dirname, filename)

        test_imge_list.append(a)
test_imge_list
for image in test_imge_list:

    show_image_label(image)