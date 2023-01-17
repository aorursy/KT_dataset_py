# import libraries 

import numpy as np

import matplotlib.pyplot as plt 

import PIL



# set global property for figure size

plt.rcParams['figure.figsize'] = [10, 8]
# image converter to tensor 

def image_reader(img_path):

    # read image in grayscale

    image = PIL.Image.open(img_path).convert('RGB').convert('L')

    # return nparray converted image

    return np.array(image)



# function to show image 

def plot_image(image, title):

    plt.title(title)

    plt.imshow(image, cmap='gray')

    plt.show()
# get image. If you want to try a mug, try it! 

path_mug = '../input/imagerecogsample/8012801_R_Z001A_UC1268924.jpg'

path_river = '../input/imagerecogsample/download.jpg'

image = image_reader(path_river)
# check image shape 

image.shape
# show original image

plot_image(image, 'Original image')
# image preprocessing

def image_preprocess(image):

    left = np.array([[0]] * image.shape[0])

    right = left.copy()

    upper = np.expand_dims(np.zeros(image.shape[1] + 2), axis=0)

    lower = upper.copy()

    

    image = np.append(image, right, axis=1)

    image = np.append(left, image, axis=1)

    image = np.append(image, lower, axis=0)

    image = np.append(upper, image, axis=0)

    

    return image
'''

Average filtering

'''



# create filter for average filtering

def average_filter(size):

    return np.ones((size, size)) / (size**2)



# simple filters in image

def simple_filter_image(image, filter_size):

    # create result numpy array with zeros 

    filtered = np.zeros(image.shape)

    

    # add padding to image data  

    image = image_preprocess(image)

    

    # get filter from function defined 

    filter_ = average_filter(filter_size)

    

    # update filter_size from width of kernel to number of pixels from center to edge of kernel

    filter_size = int((filter_size - 1) / 2)

    

    # filter image for every pixels 

    for i in range(1, image.shape[0]-1):

        for j in range(1, image.shape[1]-1):

            # initiate result value and get all indices of all edges 

            res = 0

            left, right = j - filter_size, j + filter_size + 1

            upper, lower = i - filter_size, i + filter_size + 1

            

            # 

            for p, f in zip(range(upper, lower), range(filter_.shape[0])):

                res += np.dot(filter_[f], image[p][left:right])

            

            filtered[i-1][j-1] = res

    

    return filtered
# average filtered image

ave_img = simple_filter_image(image, 3)

plot_image(ave_img, 'Average filtering')
'''

Gaussian blur 

'''



# Create Gaussian filter 

def gaussian_filter(filter_size, sigma):

    # get numpy array with identical values in both directions (axis)

    y, x = np.ogrid[-filter_size:filter_size+1, -filter_size:filter_size+1]

    

    # get 3 x 3 matrix based on x and y, then use equation shown to compute filter

    filter_ = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    

    # comment out if you want to see what is going on.

#     print(f'x looks: {x}\n y looks: \n{y}\n')

#     print('x**2 looks: {0}\n y**2 looks: \n{1}\n'.format(x**2, y**2))

#     print('By the property of numpy array, x**2 + y**2 is: \n{}'.format(x**2 + y**2))

        

    return filter_



# gaussian blur function 

def gaussian_blur(image, filter_size, sigma):

    filtered = np.zeros(image.shape)

    

    # add padding to image 

    image = image_preprocess(image)

    

    filter_size = int((filter_size - 1) / 2)

    

    filter_ = gaussian_filter(filter_size, sigma)



    for i in range(1, image.shape[0]-1):

        for j in range(1, image.shape[1]-1):

            res = 0

            left, right = j - filter_size, j + filter_size + 1

            upper, lower = i - filter_size, i + filter_size + 1

            

            for p, f in zip(range(upper, lower), range(filter_.shape[0])):

                res += np.dot(filter_[f], image[p][left:right])

                

            filtered[i-1][j-1] = res

    

    return filtered
# gaussian filtered image

gauss_img = gaussian_blur(image, 3, 3)

plot_image(gauss_img, 'Gaussian filtering')
'''

bilateral filter

'''



def bilateral_filtering(image, sigma_d, sigma_r, rc=1e-8):

    # add padding to image 

    image = image_preprocess(image)

    

    # mini function to compute gaussian of datapoint 

    gaussian = lambda val, sigma: (np.exp(-0.5 * val / sigma**2))

    

    # calculate filter size with sigma_d

    filter_size = int(3*sigma_d+1)

    

    # sum up all things

    wht_sum = np.ones(image.shape) * rc

    res = image * rc

    

    for i in range(-filter_size, filter_size+1):

        for j in range(-filter_size, filter_size+1):

            # weights calculated for distance

            spatial_wght = gaussian(i**2 + j**2, sigma_d)

            

            off = np.roll(image, [i, j], axis=[0, 1])

            tw = spatial_wght * gaussian((off - image)**2, sigma_r)

            res += off*tw

            wht_sum += tw

    

    return res / wht_sum
# bilateral filtered image

bilateral_img = bilateral_filtering(image, 3, 1)

plot_image(bilateral_img, 'Bilateral filtering')
# check with same config in opencv

import cv2

plt.imshow(cv2.bilateralFilter(image, 10, 3, 1), 'gray')
# Generate edge detection kernels 

def single_edge_filter(kind='normal'):

    # normal filter

    if kind == 'normal':

        F_x = np.array([[0, 0, 0], 

                        [-1, 0, 1], 

                        [0, 0, 0]]) * 0.5

        

    # prewitt filter 

    elif kind == 'prewitt':

        F_x = np.array([[-1, 0, 1], 

                        [-1, 0, 1], 

                        [-1, 0, 1]])

    

    # sobel filter

    elif kind == 'sobel':

        F_x = np.array([[-1, 0, 1], 

                        [-2, 0, 2], 

                        [-1, 0, 1]])

    

    # return filter and transpose of it

    return (F_x, F_x.T)



# edge detection for image

def single_edge_detection(image, kind='normal', thresh=75):

    # create result numpy array

    filtered = np.zeros(image.shape)

    

    # preprocess the image

    image = image_preprocess(image)

    

    # condition for canny or other process

    if kind == 'canny':

        gauss = gaussian_filter(1, 3)

        sobel_x, sobel_y = single_edge_filter('sobel')

        filter_x, filter_y = sobel_x * gauss, sobel_y * gauss

    

    else:

        filter_x, filter_y = single_edge_filter(kind)

        

    # loop through every pixels 

    for i in range(1, image.shape[0]-1):

        for j in range(1, image.shape[1]-1):

            temp_arr = np.array([])

            left, right = j - 1, j + 2

            upper, lower = i - 1, i + 2

            

            # get 3x3 matrix from image segment

            for r in range(upper, lower):

                temp_arr = np.append(temp_arr, image[r][left:right])

            

            temp_arr = temp_arr.reshape(3, 3)

            

            # get derivative of x and y dirction

            I_x = np.sum(filter_x * temp_arr)

            I_y = np.sum(filter_y * temp_arr)

            

            # get gradient 

            grad = np.sqrt(I_x**2 + I_y**2)

            

            # check for threshold

            if grad > thresh:

                filtered[i-1][j-1] = 255

            else:

                filtered[i-1][j-1] = 0

                

    return filtered
# normal edge detection

image_mug = image_reader(path_mug)

normal_img = single_edge_detection(image_mug, 'normal', 15)

plot_image(normal_img, 'Normal edge detection')
# prewitt edge detection 

prewitt_img = single_edge_detection(image_mug, 'prewitt')

plot_image(prewitt_img, 'Prewitt edge detection')
# sobel edge detection

sobel_img = single_edge_detection(image_mug, 'sobel', 100)

plot_image(sobel_img, 'Sobel edge detection')
# canny edge detection

canny_img = single_edge_detection(image_mug, 'canny', 2)

plot_image(canny_img, 'Canny edge detection')
# double derivative kernels 

def laplacian_filter(kind='laplacian'):

    if kind == 'laplacian':

        filter_ = np.array([[0, 1, 0], 

                        [1, -4, 1], 

                        [0, 1, 0]])

    elif kind == 'diag_laplacian':

        filter_ = np.array([[1, 1, 1], 

                [1, -8, 1], 

                [1, 1, 1]])

        

    return filter_



def laplacian_edge_detection(image, kind='laplacian', thresh=20):

    # get filter

    filter_ = laplacian_filter(kind)

    

    # get result array

    filtered = np.zeros(image.shape)

    

    # preprocess image

    image = image_preprocess(image)

    

    # loop through every pixel 

    for i in range(1, image.shape[0]-1):

        for j in range(1, image.shape[1]-1):

            temp_arr = np.array([])

            left, right = j - 1, j + 2

            upper, lower = i - 1, i + 2

            

            # get matrix for small segment of image

            for r in range(upper, lower):

                temp_arr = np.append(temp_arr, image[r][left:right])

            

            temp_arr = temp_arr.reshape(3, 3)

            

            # calculate gradient

            grad = np.sum(filter_ * temp_arr)

            

            if grad > thresh:

                filtered[i-1][j-1] = 255

            else:

                filtered[i-1][j-1] = 0

                

    return filtered
laplacian_img = laplacian_edge_detection(image_mug, 'laplacian', 20)

plot_image(laplacian_img, 'Laplacian edged image')
diag_laplacian_img = laplacian_edge_detection(image_mug, 'diag_laplacian', 20)

plot_image(diag_laplacian_img, 'Diagonal laplacian edged image')
# LoG filter

log_img = laplacian_edge_detection(gaussian_blur(image_mug, 1, 3), 'diag_laplacian', 1)

plot_image(log_img, 'LoG edged image')