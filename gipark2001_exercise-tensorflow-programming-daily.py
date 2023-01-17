import os

from os.path import join





hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'



hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 

                            ['1000288.jpg',

                             '127117.jpg']]



not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'

not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in

                            ['823536.jpg',

                             '99890.jpg']]



img_paths = hot_dog_paths + not_hot_dog_paths
img_paths
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from IPython.display import Image, display

a = load_img('../input/hot-dog-not-hot-dog/seefood/train/hot_dog/1000288.jpg')

type(a)
display(a)
from IPython.display import Image, display

from learntools.deep_learning.decode_predictions import decode_predictions

import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array





image_size = 224



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)





my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = read_and_prep_images(img_paths)

preds = my_model.predict(test_data)



most_likely_labels = decode_predictions(preds, top=3)
type(test_data), type(test_data[0])
for i, img_path in enumerate(img_paths):

    display(Image(img_path))

    print(most_likely_labels[i])
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_3 import *

print("Setup Complete")
# Experiment with code outside the function, then move it into the function once you think it is right



# the following lines are given as a hint to get you started

decoded = decode_predictions(preds, top=1)

print(decoded)



def is_hot_dog(preds):

    '''

    inputs:

    preds_array:  array of predictions from pre-trained model



    outputs:

    is_hot_dog_list: a list indicating which predictions show hotdog as the most likely label

    '''

    pass

    

q_1.check()
# q_1.hint()

# q_1.solution()
def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):

    pass



# Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths were created in the setup code

my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)

print("Fraction correct in small test set: {}".format(my_model_accuracy))



# checks that your function calc_accuracy works correctly

q_2.check()
#q_2.hint()

# q_2.solution()
# import the model

from tensorflow.keras.applications import VGG16





vgg16_model = ____

# calculate accuracy on small dataset as a test

vgg16_accuracy = ____



print("Fraction correct in small dataset: {}".format(vgg16_accuracy))

q_3.check()

#q_3.hint()

#q_3.solution()