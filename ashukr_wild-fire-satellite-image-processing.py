%matplotlib inline

import numpy as np

from scipy import misc

import matplotlib.pyplot as plt
from skimage import data



photo_data = misc.imread('../input/sd-3layers.jpg')



type(photo_data)

plt.figure(figsize=(15,15))

plt.imshow(photo_data)
photo_data.shape



#print(photo_data)
photo_data.size
photo_data.min(), photo_data.max()
photo_data.mean()
photo_data[150, 250]
photo_data[150, 250, 1]
#photo_data = misc.imread('./wifire/sd-3layers.jpg')

photo_data[150, 250] = 0

plt.figure(figsize=(10,10))

plt.imshow(photo_data)
photo_data = misc.imread('../input/sd-3layers.jpg')



photo_data[200:800, : ,1] = 255

plt.figure(figsize=(10,10))

plt.imshow(photo_data)
photo_data = misc.imread('../input/sd-3layers.jpg')



photo_data[200:800,:,:] = 255

plt.figure(figsize=(10,10))

plt.imshow(photo_data)
photo_data = misc.imread('../input/sd-3layers.jpg')



photo_data[200:800, :] = 0

plt.figure(figsize=(10,10))

plt.imshow(photo_data)
#import random

plt.figure(figsize=(10,10))

plt.imshow(photo_data)

plt.figure(figsize=(10,10))

plt.imshow(photo_data)
print(len(photo_data))
rows_range = np.arange(len(photo_data))

cols_range = rows_range

print(type(rows_range))
photo_data[rows_range, cols_range] = 255
plt.figure(figsize=(15,15))

plt.imshow(photo_data)
total_rows, total_cols, total_layers = photo_data.shape

#print("photo_data = ", photo_data.shape)



X, Y = np.ogrid[:total_rows, :total_cols]

#print("X = ", X.shape, " and Y = ", Y.shape)
center_row, center_col = total_rows / 2, total_cols / 2

print("center_row = ", center_row, "AND center_col = ", center_col)

print(X - center_row)

print(Y - center_col)

dist_from_center = (X - center_row)**2 + (Y - center_col)**2

print(dist_from_center)

radius = (total_rows / 2)**2

print("Radius = ", radius)

circular_mask = (dist_from_center > radius)

print(circular_mask)

print(circular_mask[1500:1700,2000:2200])
X, Y = np.ogrid[:total_rows, :total_cols]

half_upper = X < center_row # this line generates a mask for all rows above the center



half_upper_mask = np.logical_and(half_upper, circular_mask)
photo_data = misc.imread('../input/sd-3layers.jpg')

photo_data[half_upper_mask] = 255

#photo_data[half_upper_mask] = random.randint(200,255)

plt.figure(figsize=(15,15))

plt.imshow(photo_data)
photo_data = misc.imread('../input/sd-3layers.jpg')

red_mask   = photo_data[:, : ,0] < 150



photo_data[red_mask] = 0

plt.figure(figsize=(15,15))

plt.imshow(photo_data)
photo_data = misc.imread('../input/sd-3layers.jpg')

green_mask = photo_data[:, : ,1] < 150



photo_data[green_mask] = 0

plt.figure(figsize=(15,15))

plt.imshow(photo_data)
photo_data = misc.imread('../input/sd-3layers.jpg')

blue_mask  = photo_data[:, : ,2] < 150



photo_data[blue_mask] = 0

plt.figure(figsize=(15,15))

plt.imshow(photo_data)
photo_data = misc.imread('../input/sd-3layers.jpg')



red_mask   = photo_data[:, : ,0] < 150

green_mask = photo_data[:, : ,1] > 100

blue_mask  = photo_data[:, : ,2] < 100



final_mask = np.logical_and(red_mask, green_mask, blue_mask)

photo_data[final_mask] = 0

plt.figure(figsize=(15,15))

plt.imshow(photo_data)