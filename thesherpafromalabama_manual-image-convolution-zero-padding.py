# NTUST Digital Image Processing
# Douglas J. Pecot
# HW #2

# Let's begin with some simple, pre-written scipy packages (no need to reinvent the wheel!)
# We will start by loading some basic packages and the images (images provided by NTUST)

import scipy.signal
from skimage import io
import matplotlib.pyplot as plt

# Code sourced from:
# http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html

img = io.imread('../input/peppers.tif')  # load the image as grayscale

print('image matrix size: ', img.shape )     # print the size of image
#print('\n First 5 columns and rows of the image matrix: \n', img[:5,:5]*255 )

# Plot image inside notebook
plt.imshow(img, cmap=plt.cm.gray) # Will try two different smoothing filters on this one
plt.axis('off')
plt.show()

img2 = io.imread('../input/boatsnp.png', as_gray=True)  # load the image as grayscale
print('image matrix size: ', img.shape )     # print the size of image

plt.imshow(img2, cmap=plt.cm.gray) # This will be the image we denoise later
plt.axis('off')
plt.show()
# Median filter via Scipy

med_img = scipy.signal.medfilt2d(img2, kernel_size=5)
plt.imshow(med_img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
# Gaussian Image Filter via Scipy

gaus_img = scipy.ndimage.filters.gaussian_filter(img, 2)
plt.imshow(gaus_img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
# Mean "Standard Average" filter (will "handwrite" this one)
# Will need to define some functions to get moving
# Code sourced from:
# http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html

# 12-6-18 Fixed a bug that crashed custom kernel sizes not equal to 3x3. Should work for any kernel dimensions :)

def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    
    # Add zero padding to the input image 
    image_padded = np.zeros((image.shape[0] + (kernel.shape[0]-1), 
                             image.shape[1] + (kernel.shape[1]-1)))   
    image_padded[(kernel.shape[0]//2):-(kernel.shape[0]//2), 
                 (kernel.shape[1]//2):-(kernel.shape[1]//2)] = image
    
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
    return output



import numpy as np
from functools import reduce # Need this to get number of elements in kernel


# Define a custom Kernel size and elems here
#kernel_dim = int(input("Please input kernel dimensions: ")) # Custom k-dim
#kernel = [[int(input()) for j in range(kernel_dim)] for i in range(kernel_dim)] # custom filter, make here

# Below is the actual "standard average" filter
kernel = np.ones([7,7])  
k_elements = reduce(lambda x, y: x * y, np.shape(kernel))
kernel = kernel/k_elements
cnvlvd_img = convolve2d(img,kernel)
print('\n First 5 columns and rows of the standar average matrix: \n', cnvlvd_img[:5,:5]*255)

# Plot the filtered image
plt.imshow(cnvlvd_img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
# Gaussian smoothing (time to get dirty!)
# based on information from the following:
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
# https://matthew-brett.github.io/teaching/smoothing_intro.html

# Kernel Gennerator here
def gauskernel(n, sigma): # Note, 'n'*2+1 = kernel shape, sigma is the standard deviation
    x = np.arange(-n, n+1, 1)
    y = np.arange(-n, n+1, 1)
    x2d, y2d = np.meshgrid(x, y)
    kernel = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    return kernel / (2 * np.pi * sigma ** 2) # unit integral

kernel = gauskernel(n = 3, sigma = 1)

# We will use the 2 dimensional convolution function defined earlier

cnvlvd_img = convolve2d(img,kernel)
print('\n First 5 columns and rows of the Gaussian matrix: \n', cnvlvd_img[:5,:5]*255)
# Plot the filtered image
plt.imshow(cnvlvd_img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
# Hand writing a median filter will be a bit more challenging
# We will need to look at the values inside the filter to determine
# The value to plug into the given pixel

def mediankernelconvolv(image, kernel):
    
    output = np.zeros_like(image)            # convolution output
    
    # Add zero padding to the input image 
    image_padded = np.zeros((image.shape[0] + (kernel.shape[0]-1), 
                             image.shape[1] + (kernel.shape[1]-1)))   
    image_padded[(kernel.shape[0]//2):-(kernel.shape[0]//2), 
                 (kernel.shape[1]//2):-(kernel.shape[1]//2)] = image
    
    
    
    #np.reshape(array, -1) # converts to 1D array
    
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x] = np.median(image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]])
    return output

kernel = np.ones([3,3])
kdim = [3,3] # Have to use a slightly different format for kernel than prev examples 

cnvlvd_img = mediankernelconvolv(med_img, kernel)
print('\n First 5 columns and rows of the Median matrix: \n', cnvlvd_img[:5,:5]*255)
# Plot the filtered image
plt.imshow(cnvlvd_img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
