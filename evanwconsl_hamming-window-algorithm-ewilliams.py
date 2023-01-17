# Global Wheat Head Challenge

# Evan J. Williams evanwconsl@gmail.com Princeton, NJ

# July 27, 2020

#

# Hamming Window Algorithm

# This code using a Hamming window (frequency calculation) to determine the most probable

# Areas for a Wheat Head flower.

#

# NOTE: I did not have enough time to properly train the Hamming window so I have

#       returned a partially operational test set with a confidence of 1 as default.

#

#

# 1. Install prerequisites: opencv-python, pandas, numpy, Tk, Tcl

#

# TRAINING

# 2. Set Paths of Images

# 3. Set Path of Train.csv

# 4. Run Code; check results.

#

# TEST

# 1. Set Paths of Images

# 2. Run Code

# 3. Find images with bounding boxes and test_results.csv in output directory

#

# Code inspired by https://github.com/Crop-Phenomics-Group/Leaf-GP

# https://plantmethods.biomedcentral.com/articles/10.1186/s13007-017-0266-3
#!pip install opencv-python
#!pip install pandas
from io import StringIO
import cv2 as cv
import os

import io
%matplotlib inline

from pathlib import Path

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pylab as plt



HOME = Path('.').resolve()  # Adapt this to your system

print(HOME)
# Source code is adapted from the Leaf-GP.

#

###################################################################

#                      Document History                           #

#                                                                 #

#    Author: Dr Ji Zhou (EI/JIC)                                  #         

#               <ji.zhou@tgac.ac.uk>                              # 

#            Dr Christopher Applegate (The Zhou lab, EI)          #

#               <Christopher.Applegate@earlham.ac.uk>             #

#                                                                 #

#    Date: May 2017, Version: 1.18 on TGAC internal Github        #

#    Changes: 1) Hanlde wheat images series                       #

#             3) profiled code for distribution                   #

#             4) optimsed for paralle computing libraries         #

#                                                                 #

###################################################################



#STEP 1: Import libraries 

%matplotlib inline



# Essential libraries 

import numpy as np

import scipy as sp



# Computer vision libraries 

from scipy import ndimage

from skimage.color import rgb2gray

from skimage import io 

# All other skimage functions will be listed with the code fragments below

import cv2

from matplotlib import pyplot as plt # Plotting and generating figures



# Other libraries

import math # Feature measures 

import os # Access the file system

import gc # Garbage collection 

import csv # For results output

import sys # for operating systems 

gc.enable()
OS_Delimiter = '/' 
##STEP 2.2: Reassemble the directory of the selected image 

import glob # Find all the pathnames matching a specified pattern

fullname = '../input/global-wheat-detection/test/'

# Get the full path of the image

fullPath = fullname.split("/", -1) 

# The following can handle different platforms

print('The selected image: ', fullname)

print('The full path: ', fullname)



workingRootFolder = fullname

# Locate the image directoriy 

print("The working directory: ",workingRootFolder)



workDirFolders = workingRootFolder
# Get the full path of the .CSV file

#@csvPath = csvname.split("/", -1) 

# The following can handle different platforms

#Platform_Delimiter = PlatformDelimiter()

#csvPathRef = Platform_Delimiter.join(csvPath[: -1]) + Platform_Delimiter

#print('The selected file: ', csvname)

#print('The full path: ', csvPathRef)
# Load the data

#train_df = pd.read_csv(csvname)
# Function_1 

# Return the current date and time based on the OS

import time

from datetime import datetime, timedelta



def LocalTimeStamp(): 

    """Detect curret data and time"""

    # Current OS date and time 

    currentDate = time.localtime(time.time())

    return currentDate
ourTime = LocalTimeStamp()

print(ourTime)
##STEP 2.4: Set up a result folder to contain the processing results  

# Sort different image datasets included in the GUI system 



# Get the pre-processing date 

timeStamp = LocalTimeStamp()

curYear = timeStamp[0]

curMonth = timeStamp[1]

curDay = timeStamp[2]



# Assemble a result folder for processed results

Result_Folder =  'Processed_%d' %curYear + '-%d' %curMonth + '-%d' %curDay

Result_Directory = workingRootFolder + Result_Folder



# Folder for processed results 

print('Result folder: ', Result_Directory)
#STEP 4: Start to loop through the images   

# The following libraries are used for image processing

from skimage import color

from skimage import filters

from skimage import img_as_float, img_as_ubyte

from skimage import feature

from skimage import exposure

from skimage.transform import rescale

from skimage.morphology import skeletonize

from skimage import measure

from skimage.measure import label, find_contours

from skimage.measure import regionprops

from skimage.morphology import dilation, erosion, remove_small_objects

from skimage.morphology import disk, remove_small_holes

from skimage.morphology import convex_hull_image

import matplotlib.patches as mpatches

from skimage.draw import circle





##STEP 4.1: Set up a result folder to contain the processing results  

# pattern match jpg and png files, make sure images are .jpg, .jpeg, or .png

imageTypes = ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG') 



ImgDirectory = workingRootFolder

#print(os.listdir(ImgDirectory))

ImageFiles = []

# Only select jpg related images 

for imgType in imageTypes:

    ImageFiles.extend(glob.glob(ImgDirectory + imgType))    

# Sort the image files based on their create dates 

ImageFiles.sort(key=str.lower) # changed from os.path.getmtime
number_of_images = len(ImageFiles)

print(number_of_images)
import cv2 as cv
def freqmean(crop_img,sample_size):

    """

    function returns a new square based on the variation of color in the image

    requires the input image grid and the sample size

    """

    theShape = crop_img.shape

    rows = theShape[0]

    cols = theShape[1]

    rowcount = sample_size

    colcount = sample_size

    blank_cropped = np.zeros((rows,cols,3), np.uint8)

    if rows < sample_size or cols < sample_size:

        return blank_cropped

    channels = cv.mean(crop_img)

    distance = [0,0,0]

    normalized_channels = [0,0,0]

    override = 0

    for row in range(0,rowcount):

        crop_img_row = crop_img[row]

        for col in range(0,colcount):

            crop_img_item = crop_img_row[col]

            for bgri in range(0,3):

                color_frequency = abs(crop_img_item[bgri]-channels[bgri])

                distance[bgri] = distance[bgri] + color_frequency

                # Normalize Distance

                override = 0

                distance[bgri] = np.log(distance[bgri] / channels[bgri])

                if override == 1:

                    new_value = 255

                elif override == 2:

                    new_value = 0

                else:

                    if distance[bgri] <= 0.5:

                        #new_value = channels[bgri]

                        new_value = 0

                        override = 2

                    else:

                        if distance[bgri] >= 1.1:

                            new_value = 255

                            override = 1

                            #new_value = channels[bgri] * distance[bgri]

                        else:

                            new_value = channels[bgri] 

                            if new_value > 100:

                                override = 1

                normalized_channels[bgri] = new_value

    for row in range(0,4):

        for col in range(0,4):

            for bgri in range(0,3):

                try:

                    if override == 1:

                        blank_cropped[row][col][bgri] = 255

                    elif override == 2:

                        blank_cropped[row][col][bgri] = 0

                    else:

                        blank_cropped[row][col][bgri] = 0

                        #blank_cropped[row][col][bgri] = normalized_channels[bgri]

                except:

                    blank_cropped[row][col][bgri] = 0

    return blank_cropped
def getIntensity(crop_img):

    """

    function returns the intensity of the image.

    """

    channels = cv2.mean(crop_img)

    red = channels[2]

    green = channels[1]

    blue = channels[0]

    total_intensity = red + green + blue / 3

    return total_intensity
def sparsemean(crop_img,sample_size):

    """

    function returns a new square based on the variation of color in the image

    requires the input image grid and the sample size

    sample size is repeated throughout grid.

    """

    theShape = crop_img.shape

    rows = theShape[0]

    cols = theShape[1]

    rowcount = sample_size

    colcount = sample_size

    blank_cropped = np.zeros((rows,cols,3), np.uint8)

    if rows < sample_size or cols < sample_size:

        return blank_cropped

    channels = cv2.mean(crop_img)

    distance = [0,0,0]

    normalized_channels = [0,0,0]

    override = 0

    number_of_samples_rows = int(rows/rowcount)

    for row_sample_no in range(0,number_of_samples_rows):

        start_row_sample = row_sample_no * rowcount

        end_row_sample = row_sample_no * (rowcount + 1) - 1

        for row in range(start_row_sample,end_row_sample):

            crop_img_row = crop_img[row]

            number_of_samples_cols = int(cols/colcount)

            for col_sample_no in range(0,number_of_samples_cols):

                start_col_sample = col_sample_no * colcount

                end_col_sample = col_sample_no * (colcount + 1) - 1

                for col in range(start_col_sample,end_col_sample):

                    crop_img_item = crop_img_row[col]

                    for bgri in range(0,3):

                        color_frequency = abs(crop_img_item[bgri]-channels[bgri])

                        distance[bgri] = distance[bgri] + color_frequency

                        # Normalize Distance

                        override = 0

                        distance[bgri] = np.log(distance[bgri] / channels[bgri])

                        if override == 1:

                            new_value = 255

                        elif override == 2:

                            new_value = 0

                        else:

                            if distance[bgri] <= 0.5:

                                #new_value = channels[bgri]

                                new_value = 0

                                override = 2

                            else:

                                if distance[bgri] >= 1.1:

                                    new_value = 255

                                    override = 1

                                    #new_value = channels[bgri] * distance[bgri]

                                else:

                                    new_value = channels[bgri] 

                                    if new_value > 100:

                                        override = 1

                        normalized_channels[bgri] = new_value

    for row in range(0,4):

        for col in range(0,4):

            for bgri in range(0,3):

                try:

                    if override == 1:

                        blank_cropped[row][col][bgri] = 255

                    elif override == 2:

                        blank_cropped[row][col][bgri] = 0

                    else:

                        blank_cropped[row][col][bgri] = 0

                        #blank_cropped[row][col][bgri] = normalized_channels[bgri]

                except:

                    blank_cropped[row][col][bgri] = 0

    return blank_cropped
def returnAvgIntensity(img_input,grid_size):

    """

    Function returns the average intensity of all the grids

    """

    start_point_x = 0

    start_point_y = 0

    end_point_x = start_point_x + grid_size

    end_point_y = start_point_y + grid_size

    start_point = (start_point_x,start_point_y)

    end_point = (end_point_x,end_point_y)

    width, height, depth = img_input.shape

    x_range = range(0,width,end_point_x)

    y_range = range(0,height,end_point_y)

    blank_image = np.zeros((width,height,3), np.uint8)

    blocks = []

    last_square_was_white = 0

    count_number_of_grids = 0

    runningTotal = 0

    for x in x_range:

        start_point_x = x

        if x + grid_size > width:

            end_point_x = 1024

            w = width - x

        else:

            end_point_x = start_point_x + grid_size

            w = grid_size

        for y in y_range:

            start_point_y = y

            if y + 10 > height:

                end_point_y = height

                h = height - y

            else:

                end_point_y = end_point_y + grid_size

                h = grid_size

            start_point = (start_point_x,start_point_y)

            end_point = (end_point_x,end_point_y)

            crop_img = img_input[y:y+h, x:x+w]

            #blank = freqmean(crop_img,sample_size)

            runningTotal += getIntensity(crop_img)

            count_number_of_grids += 1

    totalIntensity = runningTotal / count_number_of_grids

    return totalIntensity
def clearLowIntensity(img_input,grid_size):

    """

    Function to sample the image with a sample window applied over a grid.

    Returns a resampled image result in black and white.

    """

    start_point_x = 0

    start_point_y = 0

    end_point_x = start_point_x + grid_size

    end_point_y = start_point_y + grid_size

    start_point = (start_point_x,start_point_y)

    end_point = (end_point_x,end_point_y)

    width, height, depth = img_input.shape

    x_range = range(0,width,end_point_x)

    y_range = range(0,height,end_point_y)

    blank_image = np.zeros((width,height,3), np.uint8)

    blocks = []

    last_square_was_white = 0

    avgIntensity = returnAvgIntensity(img_input,grid_size)

    for x in x_range:

        start_point_x = x

        if x + grid_size > width:

            end_point_x = 1024

            w = width - x

        else:

            end_point_x = start_point_x + grid_size

            w = grid_size

        for y in y_range:

            start_point_y = y

            if y + 10 > height:

                end_point_y = height

                h = height - y

            else:

                end_point_y = end_point_y + grid_size

                h = grid_size

            start_point = (start_point_x,start_point_y)

            end_point = (end_point_x,end_point_y)

            crop_img = img_input[y:y+h, x:x+w]

            #blank = freqmean(crop_img,sample_size)

            #blank = sparsemean(crop_img,sample_size)

            intensity = getIntensity(crop_img)

            if intensity < avgIntensity:

                red = 0

                green = 0

                blue = 0

                cv2.rectangle(blank_image,(start_point_x,start_point_y),(end_point_x,end_point_y),(red,green,blue),-1)

            else:

                blank_image[y:y+h,x:x+w] = crop_img

    return blank_image
def highlightWheatHeads(img_input,grid_size,sample_size):

    """

    Function to sample the image with a sample window applied over a grid.

    Returns a resampled image result in black and white.

    """

    start_point_x = 0

    start_point_y = 0

    end_point_x = start_point_x + grid_size

    end_point_y = start_point_y + grid_size

    start_point = (start_point_x,start_point_y)

    end_point = (end_point_x,end_point_y)

    width, height, depth = img_input.shape

    x_range = range(0,width,end_point_x)

    y_range = range(0,height,end_point_y)

    blank_image = np.zeros((width,height,3), np.uint8)

    blocks = []

    last_square_was_white = 0

    for x in x_range:

        start_point_x = x

        if x + grid_size > width:

            end_point_x = 1024

            w = width - x

        else:

            end_point_x = start_point_x + grid_size

            w = grid_size

        for y in y_range:

            start_point_y = y

            if y + 10 > height:

                end_point_y = height

                h = height - y

            else:

                end_point_y = end_point_y + grid_size

                h = grid_size

            start_point = (start_point_x,start_point_y)

            end_point = (end_point_x,end_point_y)

            crop_img = img_input[y:y+h, x:x+w]

            blank = freqmean(crop_img,sample_size)

            #blank = sparsemean(crop_img,sample_size)

            #

            channels = cv.mean(blank)

            red = channels[0]

            green = channels[1]

            blue = channels[2]

            cv.rectangle(blank_image,(start_point_x,start_point_y),(end_point_x,end_point_y),(red,green,blue),-1)

    return blank_image
def findBoundingBoxes(result_img3,img_tmp,ImageName,image_file_name,forCVSOutput):

    # https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d



    import cv2 as cv

    #import numpy as np



    # read and scale down image

    # wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white

    # wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg

    #img = cv2.pyrDown(cv2.imread('2011-volvo-s60_100323431_h.jpg', cv2.IMREAD_UNCHANGED))



    # threshold image

    ret, threshed_img = cv.threshold(cv.cvtColor(result_img3, cv.COLOR_BGR2GRAY),

                    10, 255, cv.THRESH_BINARY)

    # find contours and get the external one



    contours, hier = cv.findContours(threshed_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)



    #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,

    #                cv2.CHAIN_APPROX_SIMPLE)



    # with each contour, draw boundingRect in green

    # a minAreaRect in red and

    # a minEnclosingCircle in blue

    boundingRectangles = []

    totalBoxes = 0

    for c in contours:

        # get the bounding rect

        x, y, w, h = cv.boundingRect(c)

        if w>50 and h>50 and abs(w-h)<100:

            totalBoxes += 1

            rectangle = (x,y,w,h)

            boundingRectangles.append(rectangle)

            # draw a green rectangle to visualize the bounding rect

            cv.rectangle(img_tmp, (x, y), (x+w, y+h), (266, 0, 0), 2)

            # Write the coordinates of this rectangle to the .CSV file.

            # We don't know the confidence so we put 1.0.

            predictionString = "1.0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)

            theDataLine = [image_file_name,predictionString]

            # Close the file object as the format of the csv is finished

            # Create the pandas DataFrame 

            forCVSOutput.append(theDataLine)

            

    #print(totalBoxes)

    #cv.drawContours(result_img3, contours, -1, (255, 255, 0), 1)



    #cv2.imshow("contours", result_img3)



    #cv2.imshow("contours", img)

    return totalBoxes
def paintTrainingBoxes(img_tmp,image_training_set):

    trainset = pd.DataFrame(image_training_set)

    boxes = trainset.loc[:,'bbox']

    for boxstr in boxes:

        boxstrlen = len(boxstr)

        boxstr = boxstr[1:boxstrlen-1]

        items = boxstr.split(',')

        x=items[0]

        y=items[1]

        w=items[2]

        h=items[3]

        x=x.strip()

        y=y.strip()

        w=w.strip()

        h=h.strip()

        x=int(float(x))

        y=int(float(y))

        w=int(float(w))

        h=int(float(h))

        cv.rectangle(img_tmp, (x, y), (x+w, y+h), (0,255, 0), 2)
# Loop through the entire set of images

# And create resulting images setored in the Processed Directory

#for i in range(0,number_of_images):

#for i in range(0,number_of_images):

Result_Directory = "/kaggle/working/"

forCVSOutput = []

for i in range(0,number_of_images):

    tmp_IMG_File = ImageFiles[i]

    Image_FullName = tmp_IMG_File

    # Other parts of an image name    

    ImageName_Length = len(Image_FullName)

    ImageDirectory = tmp_IMG_File[:(ImageName_Length * -1)]

    ImageName = Image_FullName[:-4]

    # Retrieve the Image Training Set

    #image_training_set = train_df[train_df.image_id == ImageName]

    # Buffer the image file to the memory

    img_tmp = cv.imread(tmp_IMG_File)

    #img = cv2.imread(tmp_IMG_File)

    Resize_Ratio = 1.0/(img_tmp.shape[0]/1024.0) # dynamically transfer the original resolution 

    image_resized = img_as_ubyte(rescale(img_tmp.copy(), Resize_Ratio)) 

    # Apply the Hamming window to highlight the wheat heads.

    #result_img4 = clearLowIntensity(img_tmp,10)

    result_img3 = highlightWheatHeads(img_tmp,10,4)

    image_file_name = os.path.basename(tmp_IMG_File)

    image_id = image_file_name[:-4]

    findBoundingBoxes(result_img3,img_tmp,ImageName,image_id,forCVSOutput)

    #paintTrainingBoxes(img_tmp,image_training_set)

    #fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(30, 20))

    #ax1.set_xlabel('Length', fontsize=16)

    #ax1.set_ylabel('Width', fontsize=16)

    #ax1.set_title('Resized Image', fontsize=16)

    #x1.imshow(img_tmp)

    figname = Result_Directory + image_file_name

    plt.imsave(figname,img_tmp)

    # Create the pandas DataFrame 

results = pd.DataFrame(forCVSOutput, columns = ['image_id', 'PredictionString']) 

results.to_csv("/kaggle/working/submission.csv",index=False)