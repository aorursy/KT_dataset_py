

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt
image = cv2.imread('/kaggle/input/operations-with-opencv/1Trump.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Store height and width of the image

height, width = image.shape[:2]



quarter_height, quarter_width = height/4, width/4



#       | 1 0 Tx |

#  T  = | 0 1 Ty |



# T is our translation matrix

T = np.float32([[1, 0, quarter_width], [0, 1,quarter_height]])



# We use warpAffine to transform the image using the matrix, T

img_translation = cv2.warpAffine(image, T, (width, height))

plt.imshow(img_translation)
image = cv2.imread('/kaggle/input/operations-with-opencv/1Trump.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



height, width = image.shape[:2]



# Divide by two to rototate the image around its centre

rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, .5)



rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))



plt.imshow( rotated_image)
rotated_image = cv2.transpose(image)



plt.imshow( rotated_image)
flipped = cv2.flip(image, 1)

plt.imshow( flipped) 
# load our input image

image = cv2.imread('/kaggle/input/operations-with-opencv/1Trump.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# Let's make our image 3/4 of it's original size

image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)

plt.imshow(image_scaled)
image = cv2.imread('/kaggle/input/operations-with-opencv/1abraham.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



smaller = cv2.pyrDown(image)

larger = cv2.pyrUp(smaller)



plt.imshow(image )
plt.imshow(smaller )
plt.imshow(larger)
import cv2

image = cv2.imread('/kaggle/input/operations-with-opencv/1Hillary.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width =image.shape[:2]



start_row, start_col = int(height * .25), int(width * .25)



end_row, end_col = int(height * .75), int(width * .75)



cropped = image[start_row:end_row , start_col:end_col]





plt.imshow(image)

plt.imshow(cropped)

image = cv2.imread('/kaggle/input/operations-with-opencv/1coffee.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#image = cv2.imread('/kaggle/input/opencv-samples-images/someshapes.jpg')



# Create a matrix of ones, then multiply it by a scaler of 100 

# This gives a matrix with same dimesions of our image with all values being 100

M = np.ones(image.shape, dtype = "uint8") * 175 

plt.imshow(image)
# We use this to add this matrix M, to our image

# Notice the increase in brightness

added = cv2.add(image, M)

plt.imshow(added)
# Likewise we can also subtract

# Notice the decrease in brightness

subtracted = cv2.subtract(image, M)

plt.imshow( subtracted)
# Making a sqare

square = np.zeros((300, 300), np.uint8)

cv2.rectangle(square, (50, 50), (250, 250), 255, -2)

plt.imshow(square)
# Making a ellipse

ellipse = np.zeros((300, 300), np.uint8)

cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)

plt.imshow(ellipse)
# Shows only where they intersect

And = cv2.bitwise_and(square, ellipse)

plt.imshow(And)
# Shows where either square or ellipse is 

bitwiseOr = cv2.bitwise_or(square, ellipse)

plt.imshow(bitwiseOr)
# Shows where either exist by itself

bitwiseXor = cv2.bitwise_xor(square, ellipse)

plt.imshow( bitwiseXor)
# Shows everything that isn't part of the square

bitwiseNot_sq = cv2.bitwise_not(square)

plt.imshow(bitwiseNot_sq)
image = cv2.imread('/kaggle/input/operations-with-opencv/1Sunflowers.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)



# Creating our 3 x 3 kernel

kernel_3x3 = np.ones((3, 3), np.float32) / 9

# We use the cv2.fitler2D to conovlve the kernal with an image 

blurred = cv2.filter2D(image, -1, kernel_3x3)

plt.imshow(blurred)

# Creating our 7 x 7 kernel

kernel_7x7 = np.ones((7, 7), np.float32) / 49



blurred2 = cv2.filter2D(image, -1, kernel_7x7)

plt.imshow( blurred2)
blur = cv2.blur(image, (3,3))

plt.imshow(blur)
# Instead of box filter, gaussian kernel

Gaussian = cv2.GaussianBlur(image, (7,7), 0)

plt.imshow(Gaussian)
# Takes median of all the pixels under kernel area and central 

# element is replaced with this median value

median = cv2.medianBlur(image, 5)

plt.imshow(median)
# Bilateral is very effective in noise removal while keeping edges sharp

bilateral = cv2.bilateralFilter(image, 9, 75, 75)

plt.imshow( bilateral)
# Parameters, after None are - the filter strength 'h' (5-10 is a good range)

# Next is hForColorComponents, set as same value as h again 

dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)



plt.imshow(dst)
image = cv2.imread('/kaggle/input/operations-with-opencv/1beatle.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
# Create our shapening kernel, we don't normalize since the 

# the values in the matrix sum to 1

kernel_sharpening = np.array([[-1,-1,-1], 

                              [-1,9,-1], 

                              [-1,-1,-1]])



# applying different kernels to the input image

sharpened = cv2.filter2D(image, -1, kernel_sharpening)



plt.imshow(sharpened)
# Load our image

image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
# Values below 127 goes to 0 (black, everything above goes to 255 (white)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.imshow(thresh1)
# Values below 127 go to 255 and values above 127 go to 0 (reverse of above)

ret,thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

plt.imshow(thresh2)
# Values above 127 are truncated (held) at 127 (the 255 argument is unused)

ret,thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)

plt.imshow(thresh3)
# Values below 127 go to 0, above 127 are unchanged  

ret,thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)

plt.imshow(thresh4)
# Resever of above, below 127 is unchanged, above 127 goes to 0

ret,thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

plt.imshow(thresh5)
# Values below 127 goes to 0 (black, everything above goes to 255 (white)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.imshow(thresh1)



image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')

image = cv2.imread('/kaggle/input/operations-with-opencv/1candy.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

# Let's define our kernel size

kernel = np.ones((5,5), np.uint8)
# Now we erode

erosion = cv2.erode(image, kernel, iterations = 1)

plt.imshow( erosion)
# Dilation

dilation = cv2.dilate(image, kernel, iterations = 1)

plt.imshow(dilation)
# Opening - Good for removing noise

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

plt.imshow(opening)
# Closing - Good for removing noise

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.imshow(closing)
image = cv2.imread('/kaggle/input/operations-with-opencv/1Trump.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)





sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
plt.title("sobel_x")

plt.imshow(sobel_x)
plt.title("sobel_y")

plt.imshow(sobel_y)
sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)

plt.imshow(sobel_OR)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

plt.imshow(laplacian)
# Canny Edge Detection uses gradient values as thresholds

# The first threshold gradient

canny = cv2.Canny(image, 50, 120)

plt.imshow(canny)
image = cv2.imread('/kaggle/input/opencv-samples-images/scan.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.imshow(image)



# Cordinates of the 4 corners of the original image

points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])



# Cordinates of the 4 corners of the desired output

# We use a ratio of an A4 Paper 1 : 1.41

points_B = np.float32([[0,0], [420,0], [0,594], [420,594]])

# Use the two sets of four points to compute 

# the Perspective Transformation matrix, M    

M = cv2.getPerspectiveTransform(points_A, points_B)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

 

warped = cv2.warpPerspective(image, M, (420,594))

 

plt.imshow(warped)
image = cv2.imread('/kaggle/input/operations-with-opencv/1ex2.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rows,cols,ch = image.shape



plt.imshow(image)

cv2.waitKey(0)



# Cordinates of the 4 corners of the original image

points_A = np.float32([[320,15], [700,215], [85,610]])



# Cordinates of the 4 corners of the desired output

# We use a ratio of an A4 Paper 1 : 1.41

points_B = np.float32([[0,0], [420,0], [0,594]])

 
# Use the two sets of four points to compute 

# the Perspective Transformation matrix, M    

M = cv2.getAffineTransform(points_A, points_B)



warped = cv2.warpAffine(image, M, (cols, rows))

 

plt.imshow(warped)