import os

import numpy as np

import pandas as pd

import pydicom as dcm

import time

import tensorflow as tf

from matplotlib import pyplot as plt
data_path = "../input/osic-pulmonary-fibrosis-progression"

train_path = data_path+"/train/"

patients = os.listdir(train_path)

def load_dcm(path):

    dcm_data = dcm.dcmread(path)

    pixels = dcm_data.pixel_array

    mean, std = np.mean(pixels), np.std(pixels)

    return mean, std

    # return (pixels-mean)/std, dcm_data.get("ImagePositionPatient")

mean = []

std = []

for directory, _, files in os.walk(train_path):

    for file in files:

        try:

            m, s = load_dcm(os.path.join(directory, file))

            mean.append(m)

            std.append(s)

        except Exception:

            print(os.path.join(directory, file))

    print(np.mean(mean))
fig, ax = plt.subplots(figsize=(12, 12))

ax.hist(mean, bins=100)
fig, ax = plt.subplots(figsize=(12, 12))

ax.hist(std, bins=100)