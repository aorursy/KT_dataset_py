# we don't need all of these packages for this exercise but many of them are anyways useful

from __future__ import absolute_import, division, print_function

import os

import json as pjson

import numpy as np

import matplotlib.pyplot as plt

from glob import glob

from PIL import Image

from skimage.measure import block_reduce

imreadli = lambda ipath: np.asarray(Image.open(ipath).resize((480,360), Image.ANTIALIAS))

from skimage.io import imread

from skimage.color import label2rgb

from skimage.segmentation import mark_boundaries

from skimage.morphology import reconstruction, binary_opening

from scipy.ndimage.morphology import binary_fill_holes

from skimage.color import rgb2hsv

import skimage.color as color

from skimage.measure import label, regionprops

from skimage.morphology import disk, binary_closing, convex_hull_image

import matplotlib.patches as mpatches

from collections import defaultdict

import pandas as pd

from scipy import stats

from scipy.ndimage.interpolation import zoom

import gc
def grid_pts(n):

    """

    Make a list of x, y points spanning [0,1] covering the space evenly with n^2 points

    """

    xx,yy = np.meshgrid(np.arange(n),np.arange(n))

    stack_arr = np.stack([xx.flatten(),yy.flatten()],1).astype(np.float32)/(n-1)

    return pd.DataFrame(stack_arr,columns = ["x","y"])

def rand_pts(n):

    """

    Make a list of x, y points spanning [0,1] covering the space randomly with n^2 points

    """

    return pd.DataFrame(np.random.rand(n**2,2),columns = ["x","y"]) # ** is the power operator in python, avoid ^

def layer_maker(seed_pts, nlayers = 2, th_layer = 0, ch_amp = 0.1):

    """

    Makes random layer structures inside of lists of points

    :th_layer: is the angle the layer is rotated

    :ch_amp: is the amplitude of the fluctuations

    """

    ly_pts = seed_pts.copy()

    nx = np.cos(th_layer)*ly_pts['x']-np.sin(th_layer)*ly_pts['y']

    ny = np.sin(th_layer)*ly_pts['x']+np.cos(th_layer)*ly_pts['y']

    wave_val = np.sin(ny*2*np.pi*nlayers)>0

    ch_val = 1+ch_amp*wave_val

    ly_pts['Volume'] = ch_val*1000

    ly_pts['CV'] = wave_val

    return ly_pts

# make sure things work   

print(repr(grid_pts(3)))

# pandas has built in plotting methods

%matplotlib inline

rand_pts(5).plot.scatter('x','y')

layer_maker(grid_pts(3),th_layer = np.pi/4)
# setup the initial tools

starting_pts = rand_pts(30)

roi_fun = lambda idf: idf[(np.abs(idf['x']-0.2)<0.2) & (np.abs(idf['y']-0.2)<0.2)]

%matplotlib inline

starting_pts.plot.scatter('x','y', title = 'All Points') # all points
sample_a = roi_fun(starting_pts)

sample_a.plot.scatter('x','y', title = 'Sample A') 



sample_b = starting_pts.copy()

sample_b['x'] = sample_b['x']*0.7

sample_b['y'] = sample_b['y']*0.7

sample_b = roi_fun(sample_b)

sample_b.plot.scatter('x','y', title = 'Sample B') 



sample_c = starting_pts.copy()

sample_c['x'] = sample_c['x']*0.4

sample_c['y'] = sample_c['y']*0.6

sample_c = roi_fun(sample_c)

sample_c.plot.scatter('x','y', title = 'Sample C') 



sample_d = starting_pts.copy()

sample_d['x'] = np.power(sample_d['x']/0.5,3)

sample_d['y'] =  np.power(sample_d['y']/0.5,3)

sample_d = roi_fun(sample_d)

sample_d.plot.scatter('x','y', title = 'Sample D')
out_pts=layer_maker(grid_pts(20))

#out_pts.plot.scatter('x','y', title = 'Sample A',s='Volume') 

out_pts.plot(kind='scatter', x='x', y='y', s=out_pts['Volume']-990)

out_pts.sample(3)
out_pts=layer_maker(rand_pts(20),th_layer=np.pi/4)

out_pts.plot(kind='scatter', x='x', y='y',s=out_pts['Volume']-990)

out_pts.sample(3)