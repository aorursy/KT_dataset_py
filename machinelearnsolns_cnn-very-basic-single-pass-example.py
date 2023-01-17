import math

import numpy as np

from numpy import linalg

import scipy.io.wavfile as wav

from scipy.optimize import curve_fit

from scipy import special

import scipy

import matplotlib.pyplot as plt

import tensorflow as tf

import wave

import csv

import random

import time

import os

import zipfile







print('library test ok')
tf.compat.v1.disable_eager_execution()
x  = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]

y  = [[1.,0.],[0.,-1.]]

x1 = tf.constant( x  , dtype=tf.float32 )

y1 = tf.constant( y , dtype=tf.float32 )

x2 = tf.reshape(x1, [1, 3, 3, 1])

y2 = tf.reshape(y1, [2, 2, 1, 1])

z = tf.nn.conv2d(x2, y2, [1, 1, 1, 1], "VALID")

zout = tf.reshape(z, [2,2])

with tf.compat.v1.Session() as sess:

  sess.run( tf.compat.v1.global_variables_initializer() )

  print(sess.run(zout))