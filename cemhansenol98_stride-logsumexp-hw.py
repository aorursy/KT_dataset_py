import torch

import torch.nn as nn

from PIL import Image

from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

import os

import math

import pandas as pd
PATH = Path('../input/mnistasjpg/null')
image = Image.open("../input/mnistasjpg/trainingSet/trainingSet/0/img_1.jpg")
plt.imshow(image, cmap='gray')
np.array(image).shape
df = pd.DataFrame(np.array(image))
df.style.set_properties().background_gradient("Greys")
filter_3x3 = np.array([[2,0,0],[0,1,1,],[1,0,1]])
np.array(df)[0:3,0:3] * filter_3x3
np.sum(np.array(df)[0:3,0:3] * filter_3x3)
def stride(df,filter_x):

    t = filter_x.shape[0]

    arr = []

    p=0

    k=0

    for i in range(0,26):

        for j in range(0,26):

            arr.append(np.sum(np.array(df)[i:i+t,j:j+t] * filter_x))

            p=p+1

            k=k+1

    return arr
c = stride(df,filter_3x3)
stride_df = pd.DataFrame(np.array(c).reshape(26,26))
stride_df.style.set_properties().background_gradient("Greys")
def stride_2(df,filter_x):

    t = filter_x.shape[0]

    arr = []

    p=0

    k=0

    for i in range(0,26,2):

        for j in range(0,26,2):

            arr.append(np.sum(np.array(df)[i:i+t,j:j+t] * filter_x))

            p=p+1

            k=k+1

    return arr
c_2 = stride_2(df,filter_3x3)
stride_2_df = pd.DataFrame(np.array(c_2).reshape(13,13))
stride_2_df.style.set_properties().background_gradient("Greys")
def stride_3(df,filter_x):

    t = filter_x.shape[0]

    arr = []

    p=0

    k=0

    for i in range(0,26,3):

        for j in range(0,26,3):

            arr.append(np.sum(np.array(df)[i:i+t,j:j+t] * filter_x))

            p=p+1

            k=k+1

    return arr
c_3 = stride_3(df,filter_3x3)
stride_3_df = pd.DataFrame(np.array(c_3).reshape(9,9))
stride_3_df.style.set_properties().background_gradient("Greys")
pd.DataFrame(filter_3x3).style.set_properties().background_gradient("Greys")
def log_sum_exp(a):    

    a = np.array(a)

    a_max = np.max(a,axis=0)

    out = np.log(np.sum(np.exp(a - a_max), axis=0))

    out += a_max

    return out
array = np.array([1,3,5,8,2])
log_sum_exp(array)
array_2 = np.array([2,3,5,12,2,7,1])
log_sum_exp(array_2)