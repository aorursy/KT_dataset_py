import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import glob

import tensorflow as tf

from tensorflow import keras

import cv2
training_labels = pd.read_csv('../input/he_challenge_data/data/train.csv')

count = len(training_labels)

print(count)
training_images = []

for i in range(count):

    image = cv2.imread('../input/he_challenge_data/data/train/'+str(i)+'.jpg')

    image = cv2.resize(image, (224,224))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    training_images.append(image)

training_images = np.array(training_images)

print(training_images.shape)