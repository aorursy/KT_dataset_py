import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy
DATADIR = "input/Standford Cars Dataset"

mat = scipy.io.loadmat('../input/cars_annos.mat')
# annos = sio.loadmat('../input/cars_annos.mat')
print(mat)
