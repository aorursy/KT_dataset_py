# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Here we will use the ResNet50 model from Keras. This model is pre-trained model in Keras and very useful for image recognition.
#Importing the necessary libraries.

import numpy as np
import keras
from keras.preprocessing import image
from keras.applications import resnet50

#Load the keras' ResNet50 model that was pre-trained against the ImageNet database.

model = resnet50.ResNet50()
#Load the image file, resizing it to 224x224 pixels(required by this model).Since no. of size should be equal to no of inputs to NN model.

img = image.load_img("../input/bay.jpg", target_size=(224,224))

#Convert the image to numpy array 
x= image.img_to_array(img)

#An image generally contains three dimensions- length, breadth, and color(RGB).
#Let's add a fourth dimension since Keras expects a list of image.

x = np.expand_dims(x, axis=0)   #axis = 0 --> rows, axis = 1 --> columns

x.ndim

#Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)
#Run the image through the deep neural network to make a prediction

predictions = model.predict(x)
#Look up the names of the predicted classes 

predicted_classes = resnet50.decode_predictions(predictions)
print("This is an image of:")

for imagenet_id, name , likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood". format(name, likelihood))
#Save the model:

model.save('my_image_model.h5')

print("Model saved to disk")