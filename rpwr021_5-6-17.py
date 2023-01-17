import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import skimage.feature


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

import keras

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D

from keras.utils import np_utils

%matplotlib inline