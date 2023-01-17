import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data , io
from skimage.filters import roberts, sobel
import skimage.io

skyline_color = skimage.io.imread("../input/image-rocket/rocket.jpg")
skyline = color.rgb2gray(skyline_color)
plt.figure(figsize= (6,6))
plt.imshow(skyline, cmap='gray')
#plt.imshow(skyline_color)

skyline_edge_roberts = roberts(skyline)
plt.figure(figsize=(8,8))
plt.imshow(skyline_edge_roberts, cmap='gray')

skyline_edge_sobel = sobel(skyline)
plt.figure(figsize=(8,8))
plt.imshow(skyline_edge_sobel, cmap='gray')
 
