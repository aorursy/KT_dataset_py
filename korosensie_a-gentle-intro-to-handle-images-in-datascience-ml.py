#Checking the version of OpenCV library

import cv2

cv2.__version__
#Problem---Load the image for preprocessing

#Solution---We will use OpenCV library



#importing opencv numpy and matplotlib library

import cv2

import numpy as np

from matplotlib import pyplot as plt



#Loading the image on system as grayscale

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_GRAYSCALE)



#For viewing the image we will use matplotlib library

plt.imshow(image, cmap="gray"), plt.axis("off")

plt.show()
#Showing the data type

type(image)
#Showing image data

image
#Showing the resolution

image.shape
#showing first element

image[0,0]
#Loading image in color

image_bgr = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_COLOR)



#Showing Pixel

image_bgr[0,0]

#Converting to RGB

image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)



#Showing Image

plt.imshow(image_rgb), plt.axis("off")

plt.show()
#Problem---Save the image for preprocessing

#Solution---We will use imwrite



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#Loading image as grayscale

image =cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_GRAYSCALE)



#Saving image

cv2.imwrite("new.jpg",image)
#Problem---Resize the size of an image

#Solution---We will use resize 



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#Loading image as color

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_COLOR)



#resizing the image to 50 x 50 pixel

image_50X50 =cv2.resize(image,(50,50))



#viewing image

plt.imshow(image_50X50), plt.axis("off")

plt.show()
#Problem---Crop the image

#Solution---We will use slicing to crop the image



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#Loading image as color

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_COLOR)



#Selecting portion of the image

image_cropped=image[:,150:]



#showiing image

plt.imshow(image_cropped) , plt.axis("off")

plt.show()
#Problem---Smooth or blur out the image

#Solution---We will use cv2 blur 



#importing libraries

import cv2 

import numpy as np

from matplotlib import pyplot as plt



#Loading image as color

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_COLOR)



#Blurring the image with 5X5 kernel

image_blurry = cv2.blur(image,(5,5))



#Showing the image

plt.imshow(image_blurry) , plt.axis("off")

plt.show()
#Bluring image

image_very_blurry=cv2.blur(image,(100,100))



#Displaying image

plt.imshow(image_very_blurry), plt.axis("off")

plt.show()
#Kernel are used in everything from sharpening the image to edge detection.

kernel = np.ones((5,5))/25.0



#showing kernel

kernel
#Manually applying a kernel to image

image_kernel = cv2.filter2D(image,-1,kernel)



#displaying image

plt.imshow(image_kernel), plt.xticks([]), plt.yticks([])

        

plt.show()
#Problem---Sharpen the image

#Solution---We will use filter2D to sharpen the image



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#Loading image as color

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_COLOR)

#Creating a kernel to sharpen

kernel = np.array([[0, -1, 0],

                   [-1, 6,-1],

                   [0, -1, 0]])

# Sharpen image

image_sharp = cv2.filter2D(image, -1, kernel)

#displaying image

plt.imshow(image_sharp),plt.axis("off")

plt.show()
#Problem---Increase the contrast of the image

#Solution---We will use Histogram Equalization



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#loading image

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_GRAYSCALE)

#Enhancing the contrast of image

image_enhanced = cv2.equalizeHist(image)



#Showing image

plt.imshow(image_enhanced,cmap="gray"),plt.axis("off")

plt.show()
#Loading image

image_bgr = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg")



#Converting the image to YUV

image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)



#Enhancing the contrast of image

image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])



#converting to RGB

image_rgb =cv2.cvtColor(image_yuv,cv2.COLOR_YUV2RGB)



#Showing image

plt.imshow(image_rgb),plt.axis("off")

plt.show()
#Problem---Isolate a color in an image

#Solution---We will use OpenCV to isolate the color



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



##Loading image

image_bgr = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg")



#Converting the image to HSV

image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)



#Defining the range of blue values in HSV

lower_blue = np.array([50,100,50])

upper_blue = np.array([130,255,255])



#creating mask

mask = cv2.inRange(image_hsv, lower_blue, upper_blue)



#masking image

image_bgr_masked =cv2.bitwise_and(image_bgr,image_bgr,mask=mask)



#converting BGR to RGB

image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)



#showing image

plt.imshow(image_rgb),plt.axis("off")

plt.show()
# Show image in grayscale

plt.imshow(mask, cmap='gray'), plt.axis("off")

plt.show()
#Problem---Binarize the image

#SOlution---We will use adaptive Thresholding 



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#loading image

image_grey = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg", cv2.IMREAD_GRAYSCALE)



#Applying adaptive thresholding

max_output_value=255

neighborhood_size=99

subtract_from_mean=10

image_binarized=cv2.adaptiveThreshold(image_grey,

                                     max_output_value,

                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

                                     cv2.THRESH_BINARY,

                                     neighborhood_size,

                                     subtract_from_mean)



#showing image

plt.imshow(image_binarized, cmap='gray'), plt.axis("off")

plt.show()
# Applying cv2.ADAPTIVE_THRESH_MEAN_C

image_mean_threshold = cv2.adaptiveThreshold(image_grey,

                                             max_output_value,

                                             cv2.ADAPTIVE_THRESH_MEAN_C,

                                             cv2.THRESH_BINARY,

                                             neighborhood_size,

                                             subtract_from_mean)

# Show image

plt.imshow(image_mean_threshold, cmap="gray"), plt.axis("off")

plt.show()
#Problem---Removing background or Isolating foreground of image

#Solution---We will mark the rectangle where we want the foreground and use GrabCut algorithm.

#outside the rectangle everything is considered as background



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#Loading image and converting it to RGB

image_bgr=cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg")

image_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)



#Specifying the dimensions of rectangle

rectangle=(0,56,256,150)



#creating initial mask

mask=np.zeros(image_rgb.shape[:2],np.uint8)



#creating temporary arrays for grabcut

bgdModel=np.zeros((1,65),np.float64)

fgdModel=np.zeros((1,65),np.float64)



#Running GrabCut

cv2.grabCut(image_rgb, #Our image

           mask, #the mask

           rectangle, #our rectangle

           bgdModel, #temp array for background

           fgdModel, #temp array for foreground

           5, #number of iterations

           cv2.GC_INIT_WITH_RECT )# Initiative using our rectangle



#creating mask where certain and likely background setting 0 and otherwise 1

mask_2 = np.where((mask==2)| (mask==0),0,1).astype('uint8')

            

#Multiplying image with new mask to subtract background

image_rgb_nobg = image_rgb*mask_2[:,:,np.newaxis]



#Showing image

plt.imshow(image_rgb_nobg), plt.axis("off")

plt.show()
#Displaying the first mask applied on image

plt.imshow(mask, cmap='gray'), plt.axis("off")

plt.show()
#Displaying the second mask applied on the image

plt.imshow(mask_2, cmap='gray'), plt.axis("off")

plt.show()
#Problem---Detect the edges of the Image

#Solution---We will use Canny edge detector



#Importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#loading image as grayscale

image_gray = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_GRAYSCALE)



#calculating median intensity

median_intensity = np.median(image_gray)



#Setting threshold to standard deviation above and below median intensity

lower_threshold=int(max(0,(1.0-0.33)*median_intensity))

upper_threshold=int(min(255,(1.0+0.33)*median_intensity))



#Applying canny edge detector 

image_canny =cv2.Canny(image_gray,lower_threshold,upper_threshold)



#Displaying image

plt.imshow(image_canny,cmap='gray'), plt.axis("off")

plt.show()
#Problem---Detect the corners of the image.

#Solution---We will use harris corner detector



#Importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#Loading image as grayscale

image_bgr = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg")

image_gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)

image_gray = np.float32(image_gray)



#Setting corner detector parameters

block_size = 2

aperture = 29

free_parameter = 0.04



#Detecting corner

detector_responses = cv2.cornerHarris(image_gray,

                                     block_size,

                                     aperture,free_parameter)



#Large corner markers

detector_responses =cv2.dilate(detector_responses,None)



#Only keeping detector responses greater than threshold marked as white

threshold =0.02

image_bgr[detector_responses > threshold*detector_responses.max()] =[255,255,255]



#Converting to grayscale

image_gray =cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)



#Showing image

plt.imshow(image_gray, cmap="gray"), plt.axis("off")

plt.show()



# Show potential corners

plt.imshow(detector_responses, cmap='gray'), plt.axis("off")

plt.show()
#USing Shi-Tomasi corner detector



#loading image

image_bgr = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg")

image_gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)



#Setting number of corners to detect

corners_to_detect = 15

minimum_quality_score = 0.05

minimum_distance= 25



#Detecting Corners

corners = cv2.goodFeaturesToTrack(image_gray,

                                 corners_to_detect,

                                 minimum_quality_score,

                                 minimum_distance

                                 )

corners = np.float32(corners)



#Drawing white circles at each corner

for corner in corners:

    x , y =corner[0]

    cv2.circle(image_bgr,(x,y),10,(255,255,255),-1)



#converting to grayscale

image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)



#Displaying image

plt.imshow(image_rgb,cmap="gray"), plt.axis("off")

plt.show()
#importing libraries

import cv2

from matplotlib import pyplot as plt  



#Loading image

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg")



#flipping image you can change the -1 value to 0 1 and -1 according to your need

flip_image = cv2.flip(image, -1)



#Displaying the image

plt.imshow(flip_image),plt.axis("off")

plt.show(),plt.axis("off")
#Problem---Convert the image into an observation for machine learning

#Solution---We will use Numpy's flatten



#loading libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#loading image as grayscale

image_gray = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_GRAYSCALE)



#Resizing the image to 10X10 pixels

image_10X10=cv2.resize(image_gray,(10,10))



#converting image data to 1-D vector

image_10X10.flatten()
#displaying the dimension of flatten dimension

image_10X10.flatten().shape
#The flatten has converted the 10X10 pixel to 1X100 pixel

image_10X10
#displaying the dimension of 10X10 image dimensions

image_10X10.shape
#displaying 10X10 pixel image

plt.imshow(image_10X10, cmap="gray"), plt.axis("off")

plt.show()
#loading image as color

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_COLOR)



#resizing to 10X10 pixel and seeing number of features in color image 

image_10X10 = cv2.resize(image,(10,10))

#Converting the image data to 1-D matrix for showing dimensions in grayscale

image_10X10.flatten().shape
#loading image as grayscale

image_gray = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_GRAYSCALE)



#Converting the image data to 1-D matrix for showing dimensions in grayscale

image_gray.flatten().shape
#loading image as color

image = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_COLOR)



#Converting the image data to 1-D matrix for showing dimensions in grayscale

image.flatten().shape
#Problem---Features based on the colors of image

#Solution---We will use mean value



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#loading image as color

image_bgr = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_COLOR)



#calculating the mean of each channel 

channels =cv2.mean(image_bgr)



#Swapping values of blue and red to make the image RGB model

observation = np.array([(channels[2], channels[1], channels[0])])



#Showing mean channel values

observation
#Showing the mean colors 

plt.imshow(observation),plt.axis("off")

plt.show()
#Problem---Create a set of features representing the colors of an image

#Solution---We will compute histogram to the features of images



#importing libraries

import cv2

import numpy as np

from matplotlib import pyplot as plt



#loading image as color

image_bgr = cv2.imread("../input/the-car-connection-picture-dataset/Acura_ILX_2016_27_17_200_24_4_70_55_181_25_FWD_5_4_4dr_EsR.jpg",cv2.IMREAD_COLOR)



#Converting to RGB

image_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)



#creating a list of feature values

features=[]



#calculating the histogram for each color channel

colors = ("r","g","b")



#For each channel ccalculate histogram and add to feature value list

for i,channel in enumerate(colors):

    histogram = cv2.calcHist([image_rgb],  #image

                           [i], #index of channel 

                           None, #mask

                           [256] , #Histogram size

                           [0,256]  #range

                           )

    features.extend(histogram)

    plt.plot(histogram,color=channel)

    plt.xlim([0,256])



#plotting the graph

plt.show()





    
#Creating a vector for an observation's  feature values

observation =np.array(features).flatten()



#Showing first 10  values of observation

observation[:10]
#Showing RGB channel value at 1 pixel 

image_rgb[0,0]