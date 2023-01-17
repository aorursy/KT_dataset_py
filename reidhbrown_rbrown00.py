# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



import random

random.seed(42)



print(os.listdir("../input"))

np.random.seed(42)



import tensorflow as tf

tf.set_random_seed(42)



# Any results you write to the current directory are saved as output.
submit = pd.read_csv("../input/test_sample_submission.csv")

masks = pd.read_csv("../input/train_masks.csv")
from zipfile import ZipFile

from PIL import Image

import PIL
test_dir = "../input/test_images/test_images"

train_dir = "../input/train_images/train_images"
test_files = [test_dir + "/" + s + "/images/" + s + ".png" for s in submit['ImageId'].unique()]

test_names = [s for s in submit['ImageId'].unique()]

train_files = [train_dir + "/" + s + "/images/" + s + ".png" for s in masks['ImageId'].unique()]

train_names = [s for s in masks['ImageId'].unique()]