# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_dir = '../input/image-classification/images/images'

val_dir = '../input/image-classification/validation/validation'
! pip install -q tf-nightly
import tensorflow as tf

from tensorflow import keras

tf.__version__
# change as you want

image_size = (180, 180)

batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="training",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
train_ds.class_names
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=val_dir,

    validation_split=0.9999,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
# put your code here 
# put your code here 
# put your code here 
# put your code here 
test_dir = '../input/image-classification/test/test/classify'

os.listdir(test_dir)
# put your code here 