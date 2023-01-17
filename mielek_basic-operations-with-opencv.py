import numpy as np

import matplotlib.pyplot as plt

import cv2
# We don't need to do this again, but it's a good habit

import cv2 



# Load an image using 'imread' specifying the path to image

image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')

plt.imshow(image)

print("printed")
# Let's print each dimension of the image



print('Height of Image:', int(image.shape[0]), 'pixels')

print('Width of Image: ', int(image.shape[1]), 'pixels')
# Load our input image

image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')

#plt.imshow(input)



# We use cvtColor, to convert to grayscale

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



plt.imshow(gray_image)

# Load our input image

image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')

#plt.imshow(input)



# BGR Values for the first 0,0 pixel

B, G, R = image[10, 50] 

print(B, G, R)

print(image.shape)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(gray_img.shape)

print(gray_img[10, 50]) 
gray_img[0, 0] 
#H: 0 - 180, S: 0 - 255, V: 0 - 255



# Load our input image

image = cv2.imread('/kaggle/input/operations-with-opencv/1elephant.jpg')



hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

plt.imshow(hsv_image)

import cv2

import numpy as np

from matplotlib import pyplot as plt

image = cv2.imread('/kaggle/input/operations-with-opencv/1coffee.jpg')

plt.imshow(image)
import cv2

import numpy as np



# We need to import matplotlib to create our histogram plots

from matplotlib import pyplot as plt



image = cv2.imread('/kaggle/input/operations-with-opencv/1coffee.jpg')



histogram = cv2.calcHist([image], [0], None, [256], [0, 256])



# We plot a histogram, ravel() flatens our image array 

plt.hist(image.ravel(), 256, [0, 256]); plt.show()



# Viewing Separate Color Channels

color = ('b', 'g', 'r')



# We now separate the colors and plot each in the Histogram

for i, col in enumerate(color):

    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])

    plt.plot(histogram2, color = col)

    plt.xlim([0,256])

    
import cv2

import numpy as np

from matplotlib import pyplot as plt

image = cv2.imread('/kaggle/input/operations-with-opencv/1Hillary.jpg')

plt.imshow(image)
import cv2

import numpy as np



# We need to import matplotlib to create our histogram plots

from matplotlib import pyplot as plt



image = cv2.imread('/kaggle/input/operations-with-opencv/1Hillary.jpg')



histogram = cv2.calcHist([image], [0], None, [256], [0, 256])



# We plot a histogram, ravel() flatens our image array 

plt.hist(image.ravel(), 256, [0, 256]); plt.show()



# Viewing Separate Color Channels

color = ('b', 'g', 'r')



# We now separate the colors and plot each in the Histogram

for i, col in enumerate(color):

    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])

    plt.plot(histogram2, color = col)

    plt.xlim([0,256])

    

plt.show()
import cv2

import numpy as np

from matplotlib import pyplot as plt

# Create a black image

image = np.zeros((512,512,3), np.uint8)





plt.imshow( image)

import cv2

import numpy as np

from matplotlib import pyplot as plt

# Draw a diagonal  line of thickness of 5 pixels

image = np.zeros((512,512,3), np.uint8)

cv2.line(image, (0,0), (511,511), (255,127,0), 5)

plt.imshow(image)
# Draw a Rectangle in

image = np.zeros((512,512,3), np.uint8)



cv2.rectangle(image, (100,100), (300,250), (127,50,127), -1)

plt.imshow( image)
image = np.zeros((512,512,3), np.uint8)



cv2.circle(image, (350, 350), 100, (15,75,50), -1) 

plt.imshow(image)
image = np.zeros((512,512,3), np.uint8)



# Let's define four points

pts = np.array( [[10,50], [400,50], [90,200], [50,500]], np.int32)



# Let's now reshape our points in form  required by polylines

pts = pts.reshape((-1,1,2))



cv2.polylines(image, [pts], True, (0,0,255), 3)

plt.imshow(image)
image = np.zeros((512,512,3), np.uint8)



cv2.putText(image,'Hello World!', (75,290), cv2.FONT_HERSHEY_COMPLEX, 2, (100,170,0), 1)

plt.imshow(image)