# adapted from https://www.kaggle.com/dansbecker/programming-in-tensorflow-and-keras



import numpy as np 

import pandas as pd

import os

import json

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from IPython.display import Image, display



image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)



def decode_predictions(preds, top=5, class_list_path=None):

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



image_paths = '../input/sample-images-for-kaggle-demos'

image_paths = [os.path.join(image_paths,filename) for filename in os.listdir(image_paths)]

image_data = read_and_prep_images(image_paths)

img_paths = image_paths



test_data = read_and_prep_images(img_paths)
# Step 1: Attach a dataset containing a pre-trained model to your kernel

# Note: To Attach dataset use 'Add Data' button near 'commit' button then search for available datset or upload own

#       when added the dataset will be in 'input' folder of workspace

# Step 2: Specify location of pre-trained model weights ('../input/')

my_model_1 = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')



# or load weights later with



my_model_1.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
# Step 1: Open the right-side menu in the kernel editor and enable internet

# Step 2: Specify location of pre-trained model weights ('imagenet')

my_model_2 = ResNet50(weights='imagenet')
my_model = my_model_1

my_preds = my_model.predict(test_data)

preds = my_preds

most_likely_labels = decode_predictions(preds, top=1, class_list_path='../input/resnet50/imagenet_class_index.json')



for i, img_path in enumerate(img_paths[:2]):

    display(Image(img_path))

    print(most_likely_labels[i])
my_model = my_model_2

my_preds = my_model.predict(test_data)

preds = my_preds

most_likely_labels = decode_predictions(preds, top=1, class_list_path='../input/resnet50/imagenet_class_index.json')



for i, img_path in enumerate(img_paths[:2]):

    display(Image(img_path))

    print(most_likely_labels[i])