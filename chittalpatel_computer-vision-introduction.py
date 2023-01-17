import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import cv2



%matplotlib inline
# Read in the image

image = mpimg.imread('/kaggle/input/introcv/images/oranges.jpg')



# Print out the image dimensions

print('Image dimensions:', image.shape)

plt.imshow(image)
# Change from color to grayscale

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



plt.imshow(gray_image, cmap='gray')
# Specific grayscale pixel values

# Pixel value at x = 400 and y = 300 



x = 200

y = 100



print(gray_image[y,x])
# 5x5 image using just grayscale, numerical values

tiny_image = np.array([[0, 20, 30, 150, 120],

                      [200, 200, 250, 70, 3],

                      [50, 180, 85, 40, 90],

                      [240, 100, 50, 255, 10],

                      [30, 0, 75, 190, 220]])



# To show the pixel grid, use matshow

plt.matshow(tiny_image, cmap='gray')
# Read in the image

image = mpimg.imread('/kaggle/input/introcv/images/rainbow_flag.jpg')



plt.imshow(image)
# Isolate RGB channels

r = image[:,:,0]

g = image[:,:,1]

b = image[:,:,2]



# The individual color channels

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

ax1.set_title('R channel')

ax1.imshow(r, cmap='gray')

ax2.set_title('G channel')

ax2.imshow(g, cmap='gray')

ax3.set_title('B channel')

ax3.imshow(b, cmap='gray')

IMG_PATH='/kaggle/input/introcv/'
image = cv2.imread(IMG_PATH+'images/pizza_bluescreen.jpg')



print('This image is:', type(image), 

      ' with dimensions:', image.shape)
image_copy = np.copy(image)



# RGB (from BGR)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)



# Display the image copy

plt.imshow(image_copy)
# Color Threshold

lower_blue = np.array([0,0,200]) 

upper_blue = np.array([50,50,255])
# Define the masked area

mask = cv2.inRange(image_copy, lower_blue, upper_blue)



# Vizualize the mask

plt.imshow(mask, cmap='gray')
# Masking the image to let the pizza show through

masked_image = np.copy(image_copy)



masked_image[mask != 0] = [0, 0, 0]



plt.imshow(masked_image)
# Loading in a background image, and converting it to RGB 

background_image = cv2.imread(IMG_PATH+'images/space_background.jpg')

background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)



# Cropping it to the right size (514x816)

crop_background = background_image[0:514, 0:816]



# Masking the cropped background so that the pizza area is blocked

crop_background[mask == 0] = [0, 0, 0]



# Displaying the background

plt.imshow(crop_background)
# Adding the two images together to create a complete image!

complete_image = masked_image + crop_background



# Displaying the result

plt.imshow(complete_image)
image = cv2.imread(IMG_PATH+'images/water_balloons.jpg')



image_copy = np.copy(image)



image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)



plt.imshow(image)
# Converting from RGB to HSV

hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)



# HSV channels

h = hsv[:,:,0]

s = hsv[:,:,1]

v = hsv[:,:,2]



f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))



ax1.set_title('Hue')

ax1.imshow(h, cmap='gray')



ax2.set_title('Saturation')

ax2.imshow(s, cmap='gray')



ax3.set_title('Value')

ax3.imshow(v, cmap='gray')
# Color selection criteria in HSV values for getting only Pink balloons

lower_hue = np.array([160,0,0]) 

upper_hue = np.array([180,255,255])

# Defining the masked area in HSV space

mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)



# masking the image

masked_image = np.copy(image)

masked_image[mask_hsv==0] = [0,0,0]



# Vizualizing the mask

plt.imshow(masked_image)
# Helper functions

import glob # library for loading images from a directory



# This function loads in images and their labels and places them in a list

# The list contains all images and their associated labels

# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list

def load_dataset(image_dir):

    

    # Populate this empty image list

    im_list = []

    image_types = ["day", "night"]

    

    # Iterate through each color folder

    for im_type in image_types:

        

        # Iterate through each image file in each image_type folder

        # glob reads in any image with the extension "image_dir/im_type/*"

        for file in glob.glob(os.path.join(image_dir, im_type, "*")):

            

            # Read in the image

            im = mpimg.imread(file)

            

            # Check if the image exists/if it's been correctly read-in

            if not im is None:

                # Append the image, and it's type (red, green, yellow) to the image list

                im_list.append((im, im_type))

    

    return im_list







## Standardizing the input images

# Resizing each image to the desired input size: 600x1100px (hxw).



## Standardizing the output

# With each loaded image, we also specify the expected output.

# For this, we use binary numerical values 0/1 = night/day.





# This function should take in an RGB image and return a new, standardized version

# 600 height x 1100 width image size (px x px)

def standardize_input(image):

    

    # Resize image and pre-process so that all "standard" images are the same size

    standard_im = cv2.resize(image, (1100, 600))

    

    return standard_im





# Examples:

# encode("day") should return: 1

# encode("night") should return: 0

def encode(label):

    

    numerical_val = 0

    if(label == 'day'):

        numerical_val = 1

    # else it is night and can stay 0

    

    return numerical_val



# using both functions above, standardize the input images and output labels

def standardize(image_list):

    

    # Empty image data array

    standard_list = []

    

    # Iterate through all the image-label pairs

    for item in image_list:

        image = item[0]

        label = item[1]

        

        # Standardize the image

        standardized_im = standardize_input(image)

        

        # Create a numerical label

        binary_label = encode(label)

        

        # Append the image, and it's one hot encoded label to the full, processed list of image data

        standard_list.append((standardized_im, binary_label))

    

    return standard_list

# Image data directories

image_dir_training = "/kaggle/input/introcv/day_night_images/training/"

image_dir_test = "/kaggle/input/introcv/day_night_images/test/"



# Load training data

IMAGE_LIST = load_dataset(image_dir_training)
image_index = 20

selected_image = IMAGE_LIST[image_index][0]

selected_label = IMAGE_LIST[image_index][1]



print(len(IMAGE_LIST))

print(selected_image.shape)



plt.imshow(selected_image)
STANDARDIZED_LIST = standardize(IMAGE_LIST)
image_num = 0

selected_image = STANDARDIZED_LIST[image_num][0]

selected_label = STANDARDIZED_LIST[image_num][1]



# Displaying image and data about it

plt.imshow(selected_image)

print("Shape: "+str(selected_image.shape))

print("Label [1 = day, 0 = night]: " + str(selected_label))
# Converting and image to HSV colorspace

# Visualizing the individual color channels



image_num = 0

test_im = STANDARDIZED_LIST[image_num][0]

test_label = STANDARDIZED_LIST[image_num][1]



# Converting to HSV

hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)



# Printing image label

print('Label: ' + str(test_label))



# HSV channels

h = hsv[:,:,0]

s = hsv[:,:,1]

v = hsv[:,:,2]



# Plotting the original image and the three channels

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))

ax1.set_title('Standardized image')

ax1.imshow(test_im)

ax2.set_title('H channel')

ax2.imshow(h, cmap='gray')

ax3.set_title('S channel')

ax3.imshow(s, cmap='gray')

ax4.set_title('V channel')

ax4.imshow(v, cmap='gray')
def avg_brightness(rgb_image):

    # Converting image to HSV

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)



    # Adding up all the pixel values in the V channel

    sum_brightness = np.sum(hsv[:,:,2])

    area = 600*1100.0  # pixels

    

    avg = sum_brightness/area

    

    return avg
# Testing average brightness levels

# As an example, a "night" image is loaded in and its avg brightness is displayed

image_num = 190

test_im = STANDARDIZED_LIST[image_num][0]



avg = avg_brightness(test_im)

print('Avg brightness: ' + str(avg))

plt.imshow(test_im)
# This function should take in RGB image input

def estimate_label(rgb_image):

    

    # Extracting average brightness feature from an RGB image 

    avg = avg_brightness(rgb_image)

        

    # Using the avg brightness feature to predict a label (0, 1)

    predicted_label = 0

    threshold = 98

    if(avg > threshold):

        # if the average brightness is above the threshold value, we classify it as "day"

        predicted_label = 1

    # else, the pred-cted_label can stay 0 (it is predicted to be "night")

    

    return predicted_label    

    
import random



# Load test data

TEST_IMAGE_LIST = load_dataset(image_dir_test)



# Standardize the test data

STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)



# Shuffle the standardized test data

random.shuffle(STANDARDIZED_TEST_LIST)
def get_misclassified_images(test_images):

    # Tracking misclassified images by placing them into a list

    misclassified_images_labels = []



    # Iterating through all the test images

    for image in test_images:



        im = image[0]

        true_label = image[1]



        predicted_label = estimate_label(im)



        # Comparing true and predicted labels 

        if(predicted_label != true_label):

            # If these labels are not equal, the image has been misclassified

            misclassified_images_labels.append((im, predicted_label, true_label))

            

    return misclassified_images_labels

MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)



# Accuracy calculations

total = len(STANDARDIZED_TEST_LIST)

num_correct = total - len(MISCLASSIFIED)

accuracy = num_correct/total



print('Accuracy: ' + str(accuracy))

print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))
image_stripes = cv2.imread(IMG_PATH+'images/stripes.jpg')

image_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_BGR2RGB)



image_solid = cv2.imread(IMG_PATH+'images/pink_solid.jpg')

image_solid = cv2.cvtColor(image_solid, cv2.COLOR_BGR2RGB)





# Displaying the images

f, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))



ax1.imshow(image_stripes)

ax2.imshow(image_solid)
# converting to grayscale to focus on the intensity patterns in the image

gray_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_RGB2GRAY)

gray_solid = cv2.cvtColor(image_solid, cv2.COLOR_RGB2GRAY)



# normalizing the image color values from a range of [0,255] to [0,1] for further processing

norm_stripes = gray_stripes/255.0

norm_solid = gray_solid/255.0



# performing a fast fourier transform and create a scaled, frequency transform image

def ft_image(norm_image):

    '''This function takes in a normalized, grayscale image

       and returns a frequency spectrum transform of that image. '''

    f = np.fft.fft2(norm_image)

    fshift = np.fft.fftshift(f)

    frequency_tx = 20*np.log(np.abs(fshift))

    

    return frequency_tx

f_stripes = ft_image(norm_stripes)

f_solid = ft_image(norm_solid)



# displaying the images

# original images to the left of their frequency transform

f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,10))



ax1.set_title('original image')

ax1.imshow(image_stripes)

ax2.set_title('frequency transform image')

ax2.imshow(f_stripes, cmap='gray')



ax3.set_title('original image')

ax3.imshow(image_solid)

ax4.set_title('frequency transform image')

ax4.imshow(f_solid, cmap='gray')

image = mpimg.imread(IMG_PATH+'images/curved_lane.jpg')



plt.imshow(image)
#Converting to gray scale

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



plt.imshow(gray, cmap='gray')
# 3x3 array for edge detection

sobel_y = np.array([[ -1, -2, -1], 

                   [ 0, 0, 0], 

                   [ 1, 2, 1]])



sobel_x = np.array([[ -1, 0, 1], 

                   [ -2, 0, 2], 

                   [ -1, 0, 1]])



# Filtering the image using filter2D 

filtered_image = cv2.filter2D(gray, -1, sobel_y)



plt.imshow(filtered_image, cmap='gray')
# Displaying sobel x filtered image

filtered_image = cv2.filter2D(gray, -1, sobel_x)



plt.imshow(filtered_image, cmap='gray')
image = cv2.imread(IMG_PATH+'images/brain_MR.jpg')



image_copy = np.copy(image)



image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)



plt.imshow(image_copy)
# Converting to grayscale for filtering

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)



# Creating a Gaussian blurred image

gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))



ax1.set_title('original gray')

ax1.imshow(gray, cmap='gray')



ax2.set_title('blurred image')

ax2.imshow(gray_blur, cmap='gray')
# High-pass filter 



# 3x3 sobel filters for edge detection

sobel_x = np.array([[ -1, 0, 1], 

                   [ -2, 0, 2], 

                   [ -1, 0, 1]])





sobel_y = np.array([[ -1, -2, -1], 

                   [ 0, 0, 0], 

                   [ 1, 2, 1]])





# Filters the orginal and blurred grayscale images using filter2D

filtered = cv2.filter2D(gray, -1, sobel_x)



filtered_blurred = cv2.filter2D(gray_blur, -1, sobel_y)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))



ax1.set_title('original gray')

ax1.imshow(filtered, cmap='gray')



ax2.set_title('blurred image')

ax2.imshow(filtered_blurred, cmap='gray')
retval, binary_image = cv2.threshold(filtered_blurred, 30, 255, cv2.THRESH_BINARY)



plt.imshow(binary_image, cmap='gray')
# gaussian, sobel, and laplacian (edge) filters



gaussian = (1/9)*np.array([[1, 1, 1],

                           [1, 1, 1],

                           [1, 1, 1]])



sobel_x= np.array([[-1, 0, 1],

                   [-2, 0, 2],

                   [-1, 0, 1]])



sobel_y= np.array([[-1,-2,-1],

                   [0, 0, 0],

                   [1, 2, 1]])



# laplacian, edge filter

laplacian=np.array([[0, 1, 0],

                    [1,-4, 1],

                    [0, 1, 0]])



filters = [gaussian, sobel_x, sobel_y, laplacian]

filter_name = ['gaussian','sobel_x', \

                'sobel_y', 'laplacian']





# performing a fast fourier transform on each filter

# and creating a scaled, frequency transform image

f_filters = [np.fft.fft2(x) for x in filters]

fshift = [np.fft.fftshift(y) for y in f_filters]

frequency_tx = [np.log(np.abs(z)+1) for z in fshift]



# displaying 4 filters

for i in range(len(filters)):

    plt.subplot(2,2,i+1),plt.imshow(frequency_tx[i],cmap = 'gray')

    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])



plt.show()
image = cv2.imread(IMG_PATH+'images/brain_MR.jpg')



image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# Converting the image to grayscale for processing

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



plt.imshow(gray, cmap='gray')
# Canny using "wide" and "tight" thresholds



wide = cv2.Canny(gray, 30, 100)

tight = cv2.Canny(gray, 200, 240)

     

# Displaying the images

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))



ax1.set_title('wide')

ax1.imshow(wide, cmap='gray')



ax2.set_title('tight')

ax2.imshow(tight, cmap='gray')
image = cv2.imread(IMG_PATH+'images/phone.jpg')



image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.imshow(image)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)



# Parameters for Canny

low_threshold = 50

high_threshold = 100

edges = cv2.Canny(gray, low_threshold, high_threshold)



plt.imshow(edges, cmap='gray')
# Hough transform parameters

rho = 1

theta = np.pi/180

threshold = 30

min_line_length = 100

max_line_gap = 5



line_image = np.copy(image) #creating an image copy to draw lines on



# Hough on the edge-detected image

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),

                        min_line_length, max_line_gap)





# Iterating over the output "lines" and drawing lines on the image copy

for line in lines:

    for x1,y1,x2,y2 in line:

        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)

        

plt.imshow(line_image)
image = cv2.imread(IMG_PATH+'images/round_farms.jpg')



image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# Gray and blur

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)



gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)



plt.imshow(gray_blur, cmap='gray')
# for drawing circles on

circles_im = np.copy(image)



circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 

                           minDist=45,

                           param1=70,

                           param2=11,

                           minRadius=30,

                           maxRadius=100)



# converting circles into expected type

circles = np.uint16(np.around(circles))



# drawing each one

for i in circles[0,:]:

    # the outer circle

    cv2.circle(circles_im,(i[0],i[1]),i[2],(0,255,0),2)

    # the center of the circle

    cv2.circle(circles_im,(i[0],i[1]),2,(0,0,255),3)

    

plt.imshow(circles_im)



print('Circles shape: ', circles.shape)
# loading in color image for face detection

image = cv2.imread(IMG_PATH+'images/multi_faces.jpg')



# converting to RBG

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20,10))

plt.imshow(image)
# converting to grayscale

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  



plt.figure(figsize=(20,10))

plt.imshow(gray, cmap='gray')
# loading in cascade classifier

face_cascade = cv2.CascadeClassifier('/kaggle/input/introcv/detector_architectures/haarcascade_frontalface_default.xml')



# running the detector on the grayscale image

faces = face_cascade.detectMultiScale(gray, 4, 6)
# printing out the detections found

print ('We found ' + str(len(faces)) + ' faces in this image')

print ("Their coordinates and lengths/widths are as follows")

print ('=============================')

print (faces)
img_with_detections = np.copy(image)   #a copy of the original image to plot rectangle detections ontop of



# looping over our detections and draw their corresponding boxes on top of our original image

for (x,y,w,h) in faces:  

    # Note: the fourth element (255,0,0) determines the color of the rectangle, 

    # and the final argument (here set to 5) determines the width of the drawn rectangle

    cv2.rectangle(img_with_detections,(x,y),(x+w,y+h),(255,0,0),5)  



plt.figure(figsize=(20,10))

plt.imshow(img_with_detections)
# Read in the image

image = cv2.imread(IMG_PATH+'images/waffle.jpg')



# Make a copy of the image

image_copy = np.copy(image)



# Change color to RGB (from BGR)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)



plt.imshow(image_copy)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

gray = np.float32(gray)



# Detecting corners 

dst = cv2.cornerHarris(gray, 2, 3, 0.04)



# Dilating corner image to enhance corner points

dst = cv2.dilate(dst,None)



plt.imshow(dst, cmap='gray')

thresh = 0.1*dst.max()



# Creating an image copy to draw corners on

corner_image = np.copy(image_copy)



# Iterating through all the corners and draw them on the image (if they pass the threshold)

for j in range(0, dst.shape[0]):

    for i in range(0, dst.shape[1]):

        if(dst[j,i] > thresh):

            # image, center pt, radius, color, thickness

            cv2.circle( corner_image, (i, j), 1, (0,255,0), 1)



plt.imshow(corner_image)
image = cv2.imread(IMG_PATH+'images/thumbs_up_down.jpg')



image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.imshow(image)
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)



# Binary thresholded image

retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)



plt.imshow(binary, cmap='gray')
# Finding contours from thresholded, binary image

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



# Drawing all contours on a copy of the original image

contours_image = np.copy(image)

contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)



plt.imshow(contours_image)
image = cv2.imread(IMG_PATH+'images/monarch.jpg')



image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.imshow(image)
# Reshaping image into a 2D array of pixels and 3 color values (RGB)

pixel_vals = image.reshape((-1,3))



# Converting to float type

pixel_vals = np.float32(pixel_vals)
# you can change the number of max iterations for faster convergence!

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)



k = 3 # k



retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)



# converting data into 8-bit values

centers = np.uint8(centers)

segmented_data = centers[labels.flatten()]



# reshaping data into the original image dimensions

segmented_image = segmented_data.reshape((image.shape))

labels_reshape = labels.reshape(image.shape[0], image.shape[1])



plt.imshow(segmented_image)
image = cv2.imread(IMG_PATH+'images/rainbow_flag.jpg')



image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.imshow(image)
level_1 = cv2.pyrDown(image)

level_2 = cv2.pyrDown(level_1)

level_3 = cv2.pyrDown(level_2)



# Displaying the images

f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,10))



ax1.set_title('original')

ax1.imshow(image)



ax2.imshow(level_1)

ax2.set_xlim([0, image.shape[1]])

ax2.set_ylim([0, image.shape[0]])



ax3.imshow(level_2)

ax3.set_xlim([0, image.shape[1]])

ax3.set_ylim([0, image.shape[0]])



ax4.imshow(level_3)

ax4.set_xlim([0, image.shape[1]])

ax4.set_ylim([0, image.shape[0]])
img_path = IMG_PATH+'images/car.png'



bgr_img = cv2.imread(img_path)

gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)



gray_img = gray_img.astype("float32")/255



plt.imshow(gray_img, cmap='gray')

plt.show()
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

# defining four filters

filter_1 = filter_vals

filter_2 = -filter_1

filter_3 = filter_1.T

filter_4 = -filter_3

filters = np.array([filter_1, filter_2, filter_3, filter_4])
# visualizing filters

fig = plt.figure(figsize=(10, 5))

for i in range(4):

    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])

    ax.imshow(filters[i], cmap='gray')

    ax.set_title('Filter %s' % str(i+1))

    width, height = filters[i].shape

    for x in range(width):

        for y in range(height):

            ax.annotate(str(filters[i][x][y]), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if filters[i][x][y]<0 else 'black')
import torch

import torch.nn as nn

import torch.nn.functional as F



# data loading and transforming

from torchvision.datasets import FashionMNIST

from torch.utils.data import DataLoader

from torchvision import transforms
# neural network with a single convolutional layer with four filters

class Net(nn.Module):

    

    def __init__(self, weight):

        super(Net, self).__init__()

        # initializing the weights of the convolutional layer to be the weights of the 4 defined filters

        k_height, k_width = weight.shape[2:]

        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)

        self.conv.weight = torch.nn.Parameter(weight)



    def forward(self, x):

        # calculates the output of a convolutional layer

        # pre- and post-activation

        conv_x = self.conv(x)

        activated_x = F.relu(conv_x)

        

        # returns both layers

        return conv_x, activated_x

    

# instantiating the model and setting the weights

weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)

model = Net(weight)



print(model)
# helper function for visualizing the output of a given layer

# default number of filters is 4

def viz_layer(layer, n_filters= 4):

    fig = plt.figure(figsize=(20, 20))

    

    for i in range(n_filters):

        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])

        # grab layer outputs

        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')

        ax.set_title('Output %s' % str(i+1))
# plotting original image

plt.imshow(gray_img, cmap='gray')



# visualizing all filters

fig = plt.figure(figsize=(12, 6))

fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)

for i in range(4):

    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])

    ax.imshow(filters[i], cmap='gray')

    ax.set_title('Filter %s' % str(i+1))



    

# converting the image into an input Tensor

gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)



# getting the convolutional layer (pre and post activation)

conv_layer, activated_layer = model(gray_img_tensor)



# visualizing the output of a conv layer

viz_layer(conv_layer)
# Adding a pooling layer of size (4, 4)

class Net(nn.Module):

    

    def __init__(self, weight):

        super(Net, self).__init__()

        k_height, k_width = weight.shape[2:]

        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)

        self.conv.weight = torch.nn.Parameter(weight)

        # defining a pooling layer

        self.pool = nn.MaxPool2d(4, 4)



    def forward(self, x):

        # calculates the output of a convolutional layer

        # pre- and post-activation

        conv_x = self.conv(x)

        activated_x = F.relu(conv_x)

        

        # applies pooling layer

        pooled_x = self.pool(activated_x)

        

        # returns all layers

        return conv_x, activated_x, pooled_x

    

weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)

model = Net(weight)



print(model)
plt.imshow(gray_img, cmap='gray')



# visualizing all filters

fig = plt.figure(figsize=(12, 6))

fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)

for i in range(4):

    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])

    ax.imshow(filters[i], cmap='gray')

    ax.set_title('Filter %s' % str(i+1))



    

# converting the image into an input Tensor

gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)



conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)



# visualizing the output of the activated conv layer

viz_layer(activated_layer)



# visualizing the output of the pooling layer

viz_layer(pooled_layer)
# The output of torchvision datasets are PILImage images of range [0, 1]. 

# We transform them to Tensors for input into a CNN



data_transform = transforms.ToTensor()



train_data = FashionMNIST(root='./data', train=True,

                                   download=True, transform=data_transform)



test_data = FashionMNIST(root='./data', train=False,

                                  download=True, transform=data_transform)





# some stats about the training and test data

print('Train data, number of images: ', len(train_data))

print('Test data, number of images: ', len(test_data))
# preparing data loaders, set the batch_size

batch_size = 16



train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)



# specifying the image classes

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# one batch of training images

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy()



# plotting the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(batch_size):

    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(images[idx]), cmap='gray')

    ax.set_title(classes[labels[idx]])
class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        

        # 1 input image channel (grayscale), 10 output channels/feature maps

        # 3x3 square convolution kernel

        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26

        # the output Tensor for one image, will have the dimensions: (10, 26, 26)

        # after one pool layer, this becomes (10, 13, 13)

        self.conv1 = nn.Conv2d(1, 10, 3)

        

        # maxpool layer

        # pool with kernel_size=2, stride=2

        self.pool = nn.MaxPool2d(2, 2)

        

        # second conv layer: 10 inputs, 20 outputs, 3x3 conv

        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11

        # the output tensor will have dimensions: (20, 11, 11)

        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down

        self.conv2 = nn.Conv2d(10, 20, 3)

        

        # 20 outputs * the 5*5 filtered/pooled map size

        self.fc1 = nn.Linear(20*5*5, 50)

        

        # dropout with p=0.4

        self.fc1_drop = nn.Dropout(p=0.4)

        

        # finally, 10 output channels (for the 10 classes)

        self.fc2 = nn.Linear(50, 10)



    def forward(self, x):

        # two conv/relu + pool layers

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))



        # this line of code is the equivalent of Flatten in Keras

        x = x.view(x.size(0), -1)

        

        # two linear layers with dropout in between

        x = F.relu(self.fc1(x))

        x = self.fc1_drop(x)

        x = self.fc2(x)

        

        # final output

        return x



net = Net()

print(net)
import torch.optim as optim



# using cross entropy whcih combines softmax and NLL loss

criterion = nn.CrossEntropyLoss()



# stochastic gradient descent with a small learning rate AND some momentum

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# Calculating accuracy before training

correct = 0

total = 0



# Iterating through test dataset

for images, labels in test_loader:



    # forward pass to get outputs

    # the outputs are a series of class scores

    outputs = net(images)



    # the predicted class from the maximum value in the output-list of class scores

    _, predicted = torch.max(outputs.data, 1)



    # counting up total number of correct labels

    # for which the predicted and true labels are equal

    total += labels.size(0)

    correct += (predicted == labels).sum()



# the accuracy

# to convert `correct` from a Tensor into a scalar, .item() is used.

accuracy = 100.0 * correct.item() / total



print('Accuracy before training: ', accuracy)
def train(n_epochs):

    

    loss_over_time = [] # to track the loss as the network trains

    

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        

        running_loss = 0.0

        

        for batch_i, data in enumerate(train_loader):



            inputs, labels = data



            # zero the parameter (weight) gradients

            optimizer.zero_grad()



            # forward pass to get outputs

            outputs = net(inputs)



            # calculating the loss

            loss = criterion(outputs, labels)



            # backward pass to calculate the parameter gradients

            loss.backward()



            # updating the parameters

            optimizer.step()



            # printing loss statistics

            # to convert loss into a scalar and add it to running_loss, we use .item()

            running_loss += loss.item()

            

            if batch_i % 1000 == 999:    # printing every 1000 batches

                avg_loss = running_loss/1000

                # printing the avg loss over the 1000 batches

                loss_over_time.append(avg_loss)

                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, avg_loss))

                running_loss = 0.0



    print('Finished Training')

    return loss_over_time

n_epochs = 25 



training_loss = train(n_epochs)
# visualizing the loss as the network trained

plt.plot(training_loss)

plt.xlabel('1000\'s of batches')

plt.ylabel('loss')

plt.ylim(0, 2.5) # consistent scale

plt.show()
# initializing tensor and lists to monitor test loss and accuracy

test_loss = torch.zeros(1)

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



# setting the module to evaluation mode

net.eval()



for batch_i, data in enumerate(test_loader):

    

    inputs, labels = data

    

    # forward pass to get outputs

    outputs = net(inputs)



    # calculating the loss

    loss = criterion(outputs, labels)

            

    # updating average test loss 

    test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))

    

    # getting the predicted class from the maximum value in the output-list of class scores

    _, predicted = torch.max(outputs.data, 1)

    

    # comparing predictions to true label

    # this creates a `correct` Tensor that holds the number of correctly classified images in a batch

    correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

    

    # calculating test accuracy for *each* object class

    # we get the scalar value of correct items for a class, by calling `correct[i].item()`

    for i in range(batch_size):

        label = labels.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))



for i in range(10):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



        

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
# obtaining one batch of test images

dataiter = iter(test_loader)

images, labels = dataiter.next()

# getting predictions

preds = np.squeeze(net(images).data.max(1, keepdim=True)[1].numpy())

images = images.numpy()



# plotting the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(batch_size):

    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(images[idx]), cmap='gray')

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx] else "red"))