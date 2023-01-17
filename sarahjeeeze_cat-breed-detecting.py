# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

#import alll required modules except ipython image will need to be imported multiple times because  the PIL function is also named image

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Image, display

from learntools.deep_learning.decode_predictions import decode_predictions

import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.image as img

from PIL import Image

from IPython.display import Image, display

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from learntools.deep_learning.exercise_1 import *

import json







# Input data files are available in the read-only "../input/" directory





import os

#from PIL import Image

from IPython.display import Image, display



#create some useful functions for viewingthe images  



#visualise convolutiosn in black and white

def visualize_conv(image, conv):

    if conv == ____: # user hasn't written code. Return to avoid exception

        return

    conv_array = np.array(conv)

    vertical_padding = conv_array.shape[0] - 1

    horizontal_padding = conv_array.shape[1] - 1

    conv_out = scale_for_display(apply_conv_to_image(conv_array, image),

                                contrast_factor=350)

    show(np.hstack([image[:-vertical_padding, :-horizontal_padding], conv_out]), False)

    

#read and prepare the images using their images paths - define chosen image width and height here - This will ammend the images so they are all the same

image_size = 224



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)



#function to decode predictions using the existing keras library



def decode_predictions(preds, top=5, class_list_path='../input/keras-pretrained-models/imagenet_class_index.json'):

  """Decodes the prediction of an ImageNet model.

  Arguments:

      preds: Numpy tensor encoding a batch of predictions.

      top: integer, how many top-guesses to return.

      class_list_path: Path to the canonical imagenet_class_index.json file

  Returns:

      A list of lists of top class prediction tuples

      `(class_name, class_description, score)`.

      One list of tuples per sample in batch input.

  Raises:

      ValueError: in case of invalid shape of the `pred` array

          (must be 2D).

  """

  if len(preds.shape) != 2 or preds.shape[1] != 1000:

    raise ValueError('`decode_predictions` expects '

                     'a batch of predictions '

                     '(i.e. a 2D array of shape (samples, 1000)). '

                     'Found array with shape: ' + str(preds.shape))

  CLASS_INDEX = json.load(open(class_list_path))

  results = []

  for pred in preds:

    top_indices = pred.argsort()[-top:][::-1]

    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]

    result.sort(key=lambda x: x[2], reverse=True)

    results.append(result)

  return results





def load_image(fname = '../input/cat-dataset/CAT_01/00000100_002.jpg'):

    '''returns array containing greyscale values for supplied file (at thumbnail size)'''

    image_color = Image.open(fname).resize((135, 188), Image.ANTIALIAS)

    image_grayscale = image_color.convert('L')

    image_array = np.asarray(image_grayscale)

    return(image_array)
os.listdir("..//input//cat-dataset//CAT_00")



file_list = []

for root, dirs, files in os.walk("..//input//cat-dataset//CAT_00"):

    for file in files:

        if file.endswith(".jpg"):    

            file_list.append(file)

            



from IPython.display import Image, display#checkout the pics



display(Image('../input/cat-dataset/CAT_01/00000100_002.jpg'))
# looking at the convolution effects on my imagegs 

from PIL import Image

#create a convolutions

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_1 import *

horizontal_line_conv = [[1, 1], 

                        [-1, -1]]



#horizontal_line_conv = [[1000,1000],[1000,0]]

ok  = load_image()

visualize_conv(ok,horizontal_line_conv )
another_conv = [[1, 5,1], 

                        [-1, 3,-1],

                       [1,5,1]]



#horizontal_line_conv = [[1000,1000],[1000,0]]

ok  = load_image()

visualize_conv(ok,another_conv )


imag = img.imread('/kaggle/input/cat-dataset/cats/CAT_01/00000298_001.jpg')

print(imag.shape)


#create a list of the file names 

filenamelist = []

for i in file_list:

    filenamelist.append('/kaggle/input/cat-dataset/cats/CAT_00/'+i)

print(filenamelist[1])



len(filenamelist)
filenamelist.append('../input/mydandy/cat.jpg')
len(filenamelist)
#from IPython.display import Image, display

from IPython.display import Image, display



img_paths = '/kaggle/input/cat-dataset/cats/CAT_00'

for i, img_path in enumerate(filenamelist[0:4]):

    display(Image(img_path))

    #print(most_likely_labels[i])
#this section  of code takes the longest - there is not a set of images that are already labelled for these cats which is why we use the existing keras library 



#use earlier functions to prepare the images

catmat = read_and_prep_images(filenamelist)



#create model using resnet weights



my_model = ResNet50(weights='imagenet')



#make predictions of the data set 



preds = my_model.predict(catmat)



#use decode_predictions  function with the already created keras lib to predict cat species and/or other items in the picture 

most_likely_labels = decode_predictions(preds, top=5)


from IPython.display import Image, display



#enumerate through the file names and print out the image and the predictions as well as likelyhood



for i, img_path in enumerate(filenamelist[10:20]):

    display(Image(img_path))

    #print(most_likely_labels[i])

    print(most_likely_labels[i])

    

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(1, 1))

from IPython.display import Image, display

for i, img_path in enumerate(filenamelist[-1:]):

    print(img_path)

    display(Image(img_path))

    #print(most_likely_labels[i])

    print(most_likely_labels[i])

    



