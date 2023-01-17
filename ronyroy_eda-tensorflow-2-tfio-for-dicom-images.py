# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# source https://colab.research.google.com/github/tensorflow/io/blob/master/docs/tutorials/dicom.ipynb#scrollTo=WodUv8O1VKmr
list(os.listdir("../input/rsna-str-pulmonary-embolism-detection"))
%%time

DATA_PATH = "../input/rsna-str-pulmonary-embolism-detection"



train = pd.read_csv(f"{DATA_PATH}/train.csv")

test = pd.read_csv(f"{DATA_PATH}/test.csv")

sample_submission = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
train.head(10)
!ls /kaggle/input/rsna-str-pulmonary-embolism-detection/train/6897fa9de148/2bfbb7fd2e8b
train.info()
# %%time

# files = folders = 0



# for _, dirnames, filenames in os.walk(f"{DATA_PATH}/train"):

#     files += len(filenames)

#     folders += len(dirnames)

# print("Total number of folders in train:{} and files in it:{}".format(folders,files))
# len(train.StudyInstanceUID.unique()),len(train.SeriesInstanceUID.unique()),len(train.SOPInstanceUID.unique())

# (7279, 7279, 1790594)
!pip install tensorflow-io
import tensorflow as tf

import tensorflow_io as tfio

from pathlib import Path

import matplotlib.pyplot as plt
DATA_PATH = Path(DATA_PATH)
image_bytes = tf.io.read_file('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/6897fa9de148/2bfbb7fd2e8b/031618cba689.dcm')
image       = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

skipped     = tfio.image.decode_dicom_image(image_bytes, on_error='skip', dtype=tf.uint8)

lossy_image = tfio.image.decode_dicom_image(image_bytes, scale='auto', on_error='lossy', dtype=tf.uint8)
type(skipped), type(skipped.numpy()), skipped.numpy() 
image.numpy().shape, lossy_image.numpy().shape
fig, axes = plt.subplots(1,3, figsize=(10,10))



axes[0].imshow(np.squeeze(image.numpy()), cmap='gray')

axes[0].set_title('image')

axes[1].imshow(np.squeeze(lossy_image.numpy()), cmap='gray')

axes[1].set_title('lossy image');

axes[2].imshow(np.squeeze(lossy_image.numpy() - image.numpy()), cmap='gray')

axes[2].set_title('diff b/w images');
fig, axes = plt.subplots(3,1, figsize=(20,20))



axes[0].imshow(np.squeeze(image.numpy()), cmap='gray')

axes[0].set_title('image')

axes[1].imshow(np.squeeze(lossy_image.numpy()), cmap='gray')

axes[1].set_title('lossy image');
np.sum(image.numpy() - lossy_image.numpy())