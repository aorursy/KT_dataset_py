# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

input_dir = '/kaggle/input/kuzushiji-recognition/'

for dirname, _, filenames in os.walk(input_dir):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# following the amazing tutorial at https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

# create a YOLOv3 Keras model and save it to file

# based on https://github.com/experiencor/keras-yolo3
# import Kiras stuff

# training set labels

# each row consists of the image ID in one column, and the unicode of the recognized Kuzushiji characters

# along with their bounding boxes in the other column

train_labels = pd.read_csv(input_dir + 'train.csv')
train_labels.head(5)