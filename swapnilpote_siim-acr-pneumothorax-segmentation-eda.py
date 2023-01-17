import numpy as np

import pandas as pd

import cv2

import json

import os

import keras

import sklearn

import pydicom

import matplotlib.pyplot as plt

from glob import glob

print(os.listdir("../input"))
img_files = glob("../input/sample images/*.dcm")

print(len(img_files))
fig, axs = plt.subplots(2, 5, figsize=(25, 10))



for ax, img in zip(axs.flatten(), img_files):

    img = pydicom.dcmread(img)

    ax.set_title("Sex {}, Age {}, {}".format(img.PatientSex, img.PatientAge, img.BodyPartExamined))

    ax.imshow(img.pixel_array, cmap=plt.cm.bone)

    ax.grid(True)



plt.show()
sample_img = pydicom.dcmread(img_files[0])

plt.figure(figsize = (10, 10))

plt.imshow(sample_img.pixel_array, cmap=plt.cm.bone)
# Extra information present inside single image

sample_img.fix_meta_info