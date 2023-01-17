# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import h5py

import numpy as np

import json

import os

from random import randint
from keras import applications

from keras.engine import topology
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input
def load_split_weights(model, model_path_pattern='model_%d.h5', memb_size=102400000):  

    """Loads weights from split hdf5 files.

    

    Parameters

    ----------

    model : keras.models.Model

        Your model.

    model_path_pattern : str

        The path name should have a "%d" wild card in it.  For "model_%d.h5", the following

        files will be expected:

        model_0.h5

        model_1.h5

        model_2.h5

        ...

    memb_size : int

        The number of bytes per hdf5 file.  

    """

    model_f = h5py.File(model_path_pattern, "r", driver="family", memb_size=memb_size)

    topology.load_weights_from_hdf5_group_by_name(model_f, model.layers)

    

    return model
model = applications.VGG19(include_top=True, weights=None)  
keras_models_dir = '../input/keras-models'

model_path_pattern = keras_models_dir + "/vgg19_weights_tf_dim_ordering_tf_kernels_%d.h5" 

model = load_split_weights(model, model_path_pattern = model_path_pattern)
def load_img_to_np(img_path, target_size=(224, 224)):

    """Loads an image file into a numpy array for preprocess_image.

    

    Parameters

    ----------

    img_path : str

        Path for image.

    target_size : (int, int)

        Height and width for the image to be resized to.

        

    Returns

    -------

    numpy.ndarray (len(shape)=4)

    

    """

    img = image.load_img(img_path, target_size=target_size)

    

    # RGB -> BGR

    img_np = np.asarray(img)[...,::-1]

    

    # reshape for preprocess_input

    return img_np.reshape(1, img_np.shape[0], img_np.shape[1], img_np.shape[2]).copy().astype(np.float32)
def decode_predictions(preds, top=5):

    """Decodes the prediction of an ImageNet model.

    # Arguments

        preds: Numpy tensor encoding a batch of predictions.

        top: integer, how many top-guesses to return.

    # Returns

        A list of lists of top class prediction tuples

        `(class_name, class_description, score)`.

        One list of tuples per sample in batch input.

    # Raises

        ValueError: in case of invalid shape of the `pred` array

            (must be 2D).

    """

    if len(preds.shape) != 2 or preds.shape[1] != 1000:

        raise ValueError('`decode_predictions` expects '

                         'a batch of predictions '

                         '(i.e. a 2D array of shape (samples, 1000)). '

                         'Found array with shape: ' + str(preds.shape))

    fpath = '../input/keras-models/imagenet_class_index.json'

    CLASS_INDEX = json.load(open(fpath))

    results = []

    for pred in preds:

        top_indices = pred.argsort()[-top:][::-1]

        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]

        result.sort(key=lambda x: x[2], reverse=True)

        results.append(result)

    return results
def make_predition(model, img_np):

    """Make predictions for an image.  

    

    Parameters

    ----------

    model : Keras.models.Model

        Your model.

    img_np : numpy.ndarray (len(shape)=4)

        Array of images as numpy arrays.

        

    Returns

    -------

    List

    """

    preds = model.predict(preprocess_input(img_np))

    return decode_predictions(preds)
kitten_file = '%s/images/kitten.jpg' % keras_models_dir

kitten_img = image.load_img(kitten_file, target_size=(224, 224))

kitten_img
kitten_np = load_img_to_np(kitten_file)

print(make_predition(model, kitten_np))
dogs_cats_dir = "../input/dogs-vs-cats-redux-kernels-edition"

img_files = [file for file in os.listdir("%s/train/" % dogs_cats_dir) if file.endswith(".jpg")]

cat_files = ['%s/train/%s' % (dogs_cats_dir, file) for file in img_files if file.startswith("cat")]

dog_files = ['%s/train/%s' % (dogs_cats_dir, file) for file in img_files if file.startswith("dog")]
dog_idx = randint(0, len(dog_files))



dog_file = dog_files[dog_idx]

dog_img = image.load_img(dog_file, target_size=(224, 224))

dog_img
dog_np = load_img_to_np(dog_file)

print(make_predition(model, dog_np))
cat_idx = randint(0, len(dog_files))

cat_file = cat_files[cat_idx]

cat_img = image.load_img(cat_file, target_size=(224, 224))

cat_img
cat_np = load_img_to_np(cat_file)

print(make_predition(model, cat_np))
def split_h5_file(src_path, dest_path_pattern='model_%d.h5', memb_size=102400000):

    """Takes an hdf5 file and makes a copy of it split into multiple files.

    

    Parameters

    ----------

    src_path : str

        The path of the source hdf5 file.

    dest_path_pattern : str

        The path pattern of the destination hdf5 files. The path pattern should have a "%d" wild card in it.  

        For "model_%d.h5", the following

        files will be expected:

            model_0.h5

            model_1.h5

            model_2.h5

    memb_size : int

        Max number of bytes for each split file.

    """

    src_f = h5py.File(src_path,'r+')

    dest_f = h5py.File(dest_path_pattern, driver="family", memb_size=memb_size)

 

    # copy items

    for (name, _) in src_f.items():

        src_f.copy(name, dest_f) 

        

    # copy attribs

    for (name, value) in src_f.attrs.items():

        dest_f.attrs.create(name, value)    

        

    dest_f.flush()

    dest_f.close()

    src_f.close()