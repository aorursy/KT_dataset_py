# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_01.jpeg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))

plt.subplot(1, 2, 1)

plt.title("Original")

plt.imshow(image)



# Create our shapening kernel, we don't normalize since the 

# the values in the matrix sum to 1

kernel_sharpening = np.array([[-1,-1,-1], 

                              [-1,9,-1], 

                              [-1,-1,-1]])



# applying different kernels to the input image

sharpened = cv2.filter2D(image, -1, kernel_sharpening)





plt.subplot(1, 2, 2)

plt.title("Image Sharpening")

plt.imshow(sharpened)



plt.show()
# Load our new image

image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_11.jpg', 0)



plt.figure(figsize=(30, 30))

plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)



# Values below 127 goes to 0 (black, everything above goes to 255 (white)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



plt.subplot(3, 2, 2)

plt.title("Threshold Binary")

plt.imshow(thresh1)



# It's good practice to blur images as it removes noise

image = cv2.GaussianBlur(image, (3, 3), 0)



# Using adaptiveThreshold

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 



plt.subplot(3, 2, 3)

plt.title("Adaptive Mean Thresholding")

plt.imshow(thresh)





_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.subplot(3, 2, 4)

plt.title("Otsu's Thresholding")

plt.imshow(th2)





plt.subplot(3, 2, 5)

# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(image, (5,5), 0)

_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.title("Guassian Otsu's Thresholding")

plt.imshow(th3)

plt.show()
image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_10.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))

plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)



# Let's define our kernel size

kernel = np.ones((5,5), np.uint8)



# Now we erode

erosion = cv2.erode(image, kernel, iterations = 1)



plt.subplot(3, 2, 2)

plt.title("Erosion")

plt.imshow(erosion)



# 

dilation = cv2.dilate(image, kernel, iterations = 1)

plt.subplot(3, 2, 3)

plt.title("Dilation")

plt.imshow(dilation)





# Opening - Good for removing noise

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

plt.subplot(3, 2, 4)

plt.title("Opening")

plt.imshow(opening)



# Closing - Good for removing noise

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.subplot(3, 2, 5)

plt.title("Closing")

plt.imshow(closing)
image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_09.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



height, width,_ = image.shape



# Extract Sobel Edges

sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)



plt.figure(figsize=(20, 20))



plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)



plt.subplot(3, 2, 2)

plt.title("Sobel X")

plt.imshow(sobel_x)



plt.subplot(3, 2, 3)

plt.title("Sobel Y")

plt.imshow(sobel_y)



sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)



plt.subplot(3, 2, 4)

plt.title("sobel_OR")

plt.imshow(sobel_OR)



laplacian = cv2.Laplacian(image, cv2.CV_64F)



plt.subplot(3, 2, 5)

plt.title("Laplacian")

plt.imshow(laplacian)



# Canny Edge Detection uses gradient values as thresholds

# The first threshold gradient

canny = cv2.Canny(image, 50, 120)



plt.subplot(3, 2, 6)

plt.title("Canny")

plt.imshow(canny)
image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_08.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



plt.subplot(1, 2, 1)

plt.title("Original")

plt.imshow(image)



# Cordinates of the 4 corners of the original image

points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])



# Cordinates of the 4 corners of the desired output

# We use a ratio of an A4 Paper 1 : 1.41

points_B = np.float32([[0,0], [420,0], [0,594], [420,594]])

 

# Use the two sets of four points to compute 

# the Perspective Transformation matrix, M    

M = cv2.getPerspectiveTransform(points_A, points_B)





warped = cv2.warpPerspective(image, M, (420,594))



plt.subplot(1, 2, 2)

plt.title("warpPerspective")

plt.imshow(warped)

    
image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_13.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



plt.subplot(2, 2, 1)

plt.title("Original")

plt.imshow(image)



# Let's make our image 3/4 of it's original size

image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)



plt.subplot(2, 2, 2)

plt.title("Scaling - Linear Interpolation")

plt.imshow(image_scaled)



# Let's double the size of our image

img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)



plt.subplot(2, 2, 3)

plt.title("Scaling - Cubic Interpolation")

plt.imshow(img_scaled)



# Let's skew the re-sizing by setting exact dimensions

img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)



plt.subplot(2, 2, 4)

plt.title("Scaling - Skewed Size")

plt.imshow(img_scaled)
image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_06.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



plt.subplot(2, 2, 1)

plt.title("Original")

plt.imshow(image)



height, width = image.shape[:2]



# Let's get the starting pixel coordiantes (top  left of cropping rectangle)

start_row, start_col = int(height * .25), int(width * .25)



# Let's get the ending pixel coordinates (bottom right)

end_row, end_col = int(height * .75), int(width * .75)



# Simply use indexing to crop out the rectangle we desire

cropped = image[start_row:end_row , start_col:end_col]





plt.subplot(2, 2, 2)

plt.title("Cropped")

plt.imshow(cropped)
image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_04.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



plt.subplot(2, 2, 1)

plt.title("Original")

plt.imshow(image)



# Creating our 3 x 3 kernel

kernel_3x3 = np.ones((3, 3), np.float32) / 9



# We use the cv2.fitler2D to conovlve the kernal with an image 

blurred = cv2.filter2D(image, -1, kernel_3x3)



plt.subplot(2, 2, 2)

plt.title("3x3 Kernel Blurring")

plt.imshow(blurred)



# Creating our 7 x 7 kernel

kernel_7x7 = np.ones((7, 7), np.float32) / 49



blurred2 = cv2.filter2D(image, -1, kernel_7x7)



plt.subplot(2, 2, 3)

plt.title("7x7 Kernel Blurring")

plt.imshow(blurred2)
# Let's load a simple image with 3 black squares

image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_02.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



plt.subplot(2, 2, 1)

plt.title("Original")

plt.imshow(image)





# Grayscale

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



# Find Canny edges

edged = cv2.Canny(gray, 30, 200)



plt.subplot(2, 2, 2)

plt.title("Canny Edges")

plt.imshow(edged)



# Finding Contours

# Use a copy of your image e.g. edged.copy(), since findContours alters the image

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



plt.subplot(2, 2, 3)

plt.title("Canny Edges After Contouring")

plt.imshow(edged)



print("Number of Contours found = " + str(len(contours)))



# Draw all contours

# Use '-1' as the 3rd parameter to draw all

cv2.drawContours(image, contours, -1, (0,255,0), 3)



plt.subplot(2, 2, 4)

plt.title("Contours")

plt.imshow(image)



# Load image and keep a copy

image = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_07.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



plt.subplot(2, 2, 1)

plt.title("Original")

plt.imshow(image)



orig_image = image.copy()





# Grayscale and binarize

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)



# Find contours 

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)



# Iterate through each contour and compute the bounding rectangle

for c in contours:

    x,y,w,h = cv2.boundingRect(c)

    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    

    plt.subplot(2, 2, 2)

    plt.title("Bounding Rectangle")

    plt.imshow(orig_image)

    

cv2.waitKey(0) 

    

# Iterate through each contour and compute the approx contour

for c in contours:

    # Calculate accuracy as a percent of the contour perimeter

    accuracy = 0.03 * cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, accuracy, True)

    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    

    plt.subplot(2, 2, 3)

    plt.title("Approx Poly DP")

    plt.imshow(image)



plt.show()

    

# Convex Hull





image = cv2.imread('/kaggle/input/opencv-samples-images/hand.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



plt.figure(figsize=(20, 20))



plt.subplot(1, 2, 1)

plt.title("Original Image")

plt.imshow(image)





# Threshold the image

ret, thresh = cv2.threshold(gray, 176, 255, 0)



# Find contours 

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    

# Sort Contors by area and then remove the largest frame contour

n = len(contours) - 1

contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]



# Iterate through contours and draw the convex hull

for c in contours:

    hull = cv2.convexHull(c)

    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)



    plt.subplot(1, 2, 2)

    plt.title("Convex Hull")

    plt.imshow(image)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/Liu_Yuanshou_03.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
