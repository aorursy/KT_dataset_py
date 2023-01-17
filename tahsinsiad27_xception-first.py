from __future__ import print_function



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import utils

import os

print(os.listdir("../input/banglalekhaiso/data/data/train"))







# Networks

from keras.preprocessing import image

from keras.applications.xception import Xception



from keras.preprocessing.image import ImageDataGenerator



# Layers

from keras.layers import Dense, Activation, Flatten, Dropout

from keras import backend as K



# Other

from keras import optimizers

from keras import losses

from keras.optimizers import SGD, Adam

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.models import load_model



# Utils

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random, glob

import os, sys, csv

import cv2

import time, datetime



# Any results you write to the current directory are saved as output.