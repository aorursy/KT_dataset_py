with open('../input/humpback-whale-identification-fluke-location/cropping.txt', 'rt') as f: data = f.read().split('\n')[:-1]
len(data) # Number of rows in the dataset
for line in data[:5]: print(line)
data = [line.split(',') for line in data]
data = [(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for p,*coord in data]
data[0] # First row of the dataset
from PIL import Image as pil_image
from PIL.ImageDraw import Draw

def read_raw_image(p):
    return pil_image.open('../input/whale-categorization-playground/train/' + p)

def draw_dot(draw, x, y):
    draw.ellipse(((x-5,y-5),(x+5,y+5)), fill='red', outline='red')

def draw_dots(draw, coordinates):
    for x,y in coordinates: draw_dot(draw, x, y)

filename,coordinates = data[0]
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
# Suppress annoying stderr output when importing keras.
import sys
old_stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
from keras.preprocessing.image import img_to_array,array_to_img
sys.stderr = old_stderr

import numpy as np
from numpy.linalg import inv
from scipy.ndimage import affine_transform

def transform_img(x, affine):
    matrix   = affine[:2,:2]
    offset   = affine[:2,2]
    x        = np.moveaxis(img_to_array(x), -1, 0) # Change to channel first
    channels = [affine_transform(channel, matrix, offset, order=1, mode='constant', cval=np.average(channel)) for channel in x]
    return array_to_img(np.moveaxis(np.stack(channels, axis=0), 0, -1)) # Back to channel last, and image format

width, height = img.size
rotation = np.deg2rad(10)
# Place the origin at the center of the image
center = np.array([[1, 0, -height/2], [0, 1, -width/2], [0, 0, 1]]) 
# Rotate
rotate = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
# Restaure the origin
decenter = inv(center)
# Combine the transformations into one
m   = np.dot(decenter, np.dot(rotate, center))
img = transform_img(img, m)
img
def coord_transform(coordinates, m):
    result = []
    for x,y in coordinates:
        y,x,_ = m.dot([y,x,1]).astype(np.int)
        result.append((x,y))
    return result

transformed_coordinates = coord_transform(coordinates, inv(m))
transformed_coordinates
transformed_box = bounding_rectangle(transformed_coordinates)
transformed_box
draw = Draw(img)
draw.rectangle(transformed_box, outline='yellow')
img