##############################################################################
#
#   This notebook demonstrates how to apply image filters (conv) and pooling 
#   using python an numpy libray.
#
################################################################################
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage import exposure
from skimage import data, img_as_float

original_image = img_as_float(data.chelsea())
#
# Image Convoluter Class
#
class ImageConvoluter(object):
    def __init__(self):
        pass
    
    def convolve2d(self, image, kernel):
        # Takes an image and a kernel and returns the convolution of them
        # Args:
        #   image: a numpy array of size [image_height, image_width].
        #   kernel: a numpy array of size [kernel_height, kernel_width].
        # Returns:
        #   a numpy array of size [image_height, image_width] (convolution output).

        ksize = kernel.shape[0]
        kpad = int((ksize - 1)/2)

        kernel = np.flipud(np.fliplr(kernel))   # Flip the kernel    
        output = np.zeros_like(image)           # convolution output


        # Add zero padding to the input image
        image_padded = np.zeros((image.shape[0] + ksize - 1, image.shape[1] + ksize - 1))
        image_padded[kpad:-kpad, kpad:-kpad] = image

        # Loop over every pixel of the image
        for x in range(image.shape[1]): # width
            for y in range(image.shape[0]): # height
                # element-wise multiplication of the kernel and the image            
                kernel_image = kernel * image_padded[y:y+ksize, x:x+ksize]            
                output[y,x] = kernel_image.sum()
        #output = np.clip(output, 0, 255)
        return output

    def maxpool2d(self, image, size = 2, stride = 2):
        h, w = image.shape
        pool_out = np.zeros((np.uint16((h-size+1)/stride), np.uint16((w-size+1)/stride)))

        r2 = 0
        for r in np.arange(0, h-size-(stride - 1), stride):  
            c2 = 0
            for c in np.arange(0, w-size-(stride - 1), stride):
                pool_out[r2, c2] = np.max(image[r:r+size,  c:c+size])  
                c2 = c2 + 1
            r2 = r2 +1    
        return pool_out
#
# Sample Filters
#
sharpen_kernel = np.array(
    [[ 0,-1, 0],
     [-1, 5,-1],
     [ 0,-1, 0]])

sharpen_kernel2 = np.array(
    [[ -1,-1,-1,-1,-1],
     [ -1,-1,-1,-1,-1],
     [ -1,-1,25,-1,-1],
     [ -1,-1,-1,-1,-1],
     [ -1,-1,-1,-1,-1]])

edge_detection_kernel = np.array(
    [[-1,-1,-1],
     [-1, 8,-1],
     [-1,-1,-1]])

blur_kernel = np.array(
    [[1/9,1/9,1/9],
     [1/9,1/9,1/9],
     [1/9,1/9,1/9]])

Gaussian_kernel = np.array([
    [1/256, 4/256, 6/256, 4/256,1/256],
    [4/256,16/256,24/256,16/256,4/256],
    [6/256,24/256,36/256,24/256,6/256],
    [4/256,16/256,24/256,16/256,4/256],
    [1/256, 4/256, 6/256, 4/256,1/256]
])

sobel_h = np.array([
    [-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2, 1]
])

sobel_v = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
#
# Show convoluted image via sample kernels
#
def show_image(image, title_text):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title_text)
    plt.show()

kernels = [(sharpen_kernel, 'Sharpen'), (sharpen_kernel2, 'Sharpen 2'), (edge_detection_kernel, 'Edge Detector'), (blur_kernel, 'Blur'), (Gaussian_kernel, 'Gaussian')]

conv = ImageConvoluter()
cat_image = color.rgb2gray(original_image) * 255
scale = 2
new_size = (int(cat_image.shape[0]/scale), int(cat_image.shape[1]/scale))
resized_cat_image = resize(cat_image, new_size)

show_image(original_image, 'Original image')
show_image(resized_cat_image, 'Original resized gray image')

pooled_resized_cat_image = conv.maxpool2d(resized_cat_image, 2, 2)
show_image(pooled_resized_cat_image, 'Pooled (2x2) resized gray image')

for (kernel, kernel_name) in kernels:    
    pooled_image = conv.maxpool2d(resized_cat_image, 2, 2)
    conv_image = conv.convolve2d(pooled_image, kernel)
    show_image(conv_image, 'pooled and ' + kernel_name)

# sober filter
pooled_image = conv.maxpool2d(resized_cat_image, 2, 2)
conv_image1 = conv.convolve2d(pooled_image, sobel_h)
conv_image2 = conv.convolve2d(pooled_image, sobel_v)
conv_image = (conv_image1 + conv_image2)/2
conv_image = conv_image.astype(int)
show_image(conv_image1, 'Sober Horizental')
show_image(conv_image2, 'Sober Vertical')
#
# Use scipy.signal for convolution
#
import scipy.signal

for (kernel, kernel_name) in kernels:    
    conv_image = scipy.signal.convolve2d(cat_image, kernel, 'valid')    
    show_image(conv_image, kernel_name)
