# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from tqdm import tqdm_notebook as tqdm
from PIL import Image as pil_image
from PIL.ImageDraw import Draw
import pandas as pd
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageDraw import Draw

TRAIN_DF = '../input/humpback-whale-identification/train.csv'
SUB_Df = '../input/humpback-whale-identification/sample_submission.csv'
TRAIN = '../input/humpback-whale-identification/train/'
TEST = '../input/humpback-whale-identification/test/'
BB_DF = "../input/metadata/bounding_boxes.csv"
tagged = dict([(p, w) for _, p, w in pd.read_csv(TRAIN_DF).to_records()])
submit = [p for _, p, _ in pd.read_csv(SUB_Df).to_records()]
join = list(tagged.keys()) + submit
def expand_path(p):
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p
# Raw data image read, if data in rotate list, 180 deg rotate
def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    #if p in rotate: img = img.rotate(180)  # if rotation is required,
    return img
def show_whale(imgs, per_row=2):
    n         = len(imgs)
    rows      = (n + per_row - 1)//per_row
    cols      = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))
    for ax in axes.flatten(): ax.axis('off')
    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))
# image size data generation

p2size = {}
for p in tqdm(join):
    size = pil_image.open(expand_path(p)).size
    p2size[p] = size
# anisotropy; The horizontal compression ratio, ~ 2

total_width = 0
total_height = 0

for val in p2size.values():
    total_width  += val[0]
    total_height += val[1]
    
avg_width = total_width/len(p2size)
avg_height = total_height/len(p2size)
ratio = avg_width/avg_height
ratio
# Bounding box data load

p2bb = pd.read_csv(BB_DF).set_index("Image")
p2bb.head()
# example
filename = list(tagged.keys())[100]
filename
def draw_dot(draw, x, y):
    draw.ellipse(((x-5,y-5),(x+5,y+5)), fill='red', outline='red')

def draw_dots(draw, coordinates):
    for x,y in coordinates: draw_dot(draw, x, y)

size_x,size_y = p2size[filename]
x0, y0, x1, y1 = p2bb.loc[filename]

#if rotation is required,
#if filename in rotate:
#    x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0

coordinates = [(x0, y0), (x1, y1)]
img = read_raw_image(filename)
draw = Draw(img)
draw_dots(draw, coordinates)
img
def bounding_rectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x,y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0,y0,x1,y1

box = bounding_rectangle(coordinates)
box
draw.rectangle(box, outline='red')
img
import sys
import platform
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
sys.stderr = old_stderr

import random
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform

img_shape    = (384,384,3) # The image shape used by the model
anisotropy   = 2.0 # The horizontal compression ratio
#crop_margin  = 0.05 # The margin added around the bounding box to compensate for bounding box inaccuracy

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation        = np.deg2rad(rotation)
    shear           = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix    = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix     = np.array([[1.0/height_zoom, 0, 0], [0, 1.0/width_zoom, 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def read_cropped_image(p, augment, crop_margin=0.05):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    size_x,size_y = p2size[p]
    
    # Determine the region of the original image we want to capture based on the bounding box.
    x0,y0,x1,y1   = p2bb.loc[p]
    # if rotation is required,
    #if p in rotate: x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0

    dx            = x1 - x0
    dy            = y1 - y0
    x0           -= dx*crop_margin
    x1           += dx*crop_margin + 1
    y0           -= dy*crop_margin
    y1           += dy*crop_margin + 1
    if (x0 < 0     ): x0 = 0
    if (x1 > size_x): x1 = size_x
    if (y0 < 0     ): y0 = 0
    if (y1 > size_y): y1 = size_y
    dx            = x1 - x0
    dy            = y1 - y0
    if dx > dy*anisotropy:
        dy  = 0.5*(dx/anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx  = 0.5*(dy*anisotropy - dx)
        x0 -= dx
        x1 += dx
    
    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5*img_shape[0]], [0, 1, -0.5*img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0)/img_shape[0], 0, 0], [0, (x1 - x0)/img_shape[1], 0], [0, 0, 1]]), trans)
    
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05*(y1 - y0), 0.05*(y1 - y0)),
            random.uniform(-0.05*(x1 - x0), 0.05*(x1 - x0))
            ), trans) 
    trans = np.dot(np.array([[1, 0, 0.5*(y1 + y0)], [0, 1, 0.5*(x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, Comvert to numpy array
    img   = read_raw_image(p)
    img   = img_to_array(img)
    
    # Apply affine transformation
    matrix = trans[:2,:2]
    offset = trans[:2,2]
    x = np.moveaxis(img, -1, 0) # Change to channel first
    
    img    = [affine_transform(img, matrix, offset, order=1, output_shape=img_shape[:-1], mode='constant', cval=np.average(img)) for img in x]
    img    = np.moveaxis(np.stack(img, axis=0), 0, -1)

    # Normalize to zero mean and unit variance
    img  -= np.mean(img, keepdims=True)
    img  /= np.std(img, keepdims=True) + K.epsilon()
    
    return img

def read_for_training(p, crop_margin=0.05):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    print("Training crop_margin: ", crop_margin)
    return read_cropped_image(p, True, crop_margin)

def read_for_validation(p, crop_margin=0.05):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    print("Validation crop_margin: ", crop_margin)
    return read_cropped_image(p, False, crop_margin)

imgs = [
    read_raw_image(filename),                                      # raw image plot
    array_to_img(read_for_validation(filename)),                   # Affine transform (resize with bbox) without augmentation
    array_to_img(read_for_validation(filename, crop_margin=0)),    # Affine transform (resize with bbox) without augmentation, margin = 0
    array_to_img(read_for_training(filename)),                     # Affine transform (resize with bbox) with augmentation
    array_to_img(read_for_training(filename, crop_margin=0))       # Affine transform (resize with bbox) with augmentation, margin = 0
]
show_whale(imgs, per_row=5)
img_shape    = (384,384,1)

def read_cropped_image(p, augment, crop_margin=0.05):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    size_x,size_y = p2size[p]
    
    # Determine the region of the original image we want to capture based on the bounding box.
    x0,y0,x1,y1   = p2bb.loc[p]
    dx            = x1 - x0
    dy            = y1 - y0
    x0           -= dx*crop_margin
    x1           += dx*crop_margin + 1
    y0           -= dy*crop_margin
    y1           += dy*crop_margin + 1
    if (x0 < 0     ): x0 = 0
    if (x1 > size_x): x1 = size_x
    if (y0 < 0     ): y0 = 0
    if (y1 > size_y): y1 = size_y
    dx            = x1 - x0
    dy            = y1 - y0
    if dx > dy*anisotropy:
        dy  = 0.5*(dx/anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx  = 0.5*(dy*anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5*img_shape[0]], [0, 1, -0.5*img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0)/img_shape[0], 0, 0], [0, (x1 - x0)/img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05*(y1 - y0), 0.05*(y1 - y0)),
            random.uniform(-0.05*(x1 - x0), 0.05*(x1 - x0))
            ), trans)
    trans = np.dot(np.array([[1, 0, 0.5*(y1 + y0)], [0, 1, 0.5*(x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img   = read_raw_image(p).convert('L')
    img   = img_to_array(img)
    
    # Apply affine transformation
    matrix = trans[:2,:2]
    offset = trans[:2,2]
    img    = img.reshape(img.shape[:-1])
    img    = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant', cval=np.average(img))
    img    = img.reshape(img_shape)
    
    # Normalize to zero mean and unit variance
    img  -= np.mean(img, keepdims=True)
    img  /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True)

def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False)

imgs = [
    read_raw_image(filename),                      # raw image plot
    array_to_img(read_for_validation(filename)),   # Affine transform (resize with bbox) without augmentation
    array_to_img(read_for_training(filename))      # Affine transform (resize with bbox) with augmentation
]
show_whale(imgs, per_row=3)
