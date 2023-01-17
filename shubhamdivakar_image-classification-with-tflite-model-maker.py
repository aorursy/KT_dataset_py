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
!pip install git+git://github.com/tensorflow/examples.git#egg=tensorflow-examples[model_maker]
import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task.model_spec import mobilenet_v2_spec
from tensorflow_examples.lite.model_maker.core.task.model_spec import ImageModelSpec

import matplotlib.pyplot as plt
#load flower dataset
image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)

#below code is for unpacking archives/zip/rar
#!pip install pyunpack
#!pip install patool
#from pyunpack import Archive
#Archive('/content/stage 01.rar').extractall('/content/new/')

#load your own dataset
#image_path= '/content/new/stage 01/'
data = ImageClassifierDataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
model = image_classifier.create(train_data,model_spec=mobilenet_v2_spec)
loss, accuracy = model.evaluate(test_data)
model.export(export_dir='.', with_metadata=True)
from IPython.display import FileLink, FileLinks
FileLinks('.') #generates links of all files
#FileLinks('model.tflite')
#FileLinks('labels.txt')