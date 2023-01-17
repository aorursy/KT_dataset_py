import os

import numpy as np

import pandas as pd

from scipy.misc import imread

import matplotlib.pyplot as plt

%pylab inline

pars = {"figure.figsize" : (20,8),

        "axes.titlesize" : 36,

        "axes.titleweight": "bold"}

pylab.rcParams.update(pars)                                         



data_dir = "../input/train"

x = imread('../input/train/BET/img_07734.jpg')

y = imread('../input/train/BET/img_07804.jpg')

print(x.shape[0])

print(y.shape[0])

import skimage.measure as sm

print(sm.structural_similarity(x, y, win_size=3,gradient=False, dynamic_range=None))