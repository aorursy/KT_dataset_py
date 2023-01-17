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
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/000_1ov3n5_0.jpeg')

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

image = cv2.imread('/kaggle/input//medical-masks-dataset/images/003_1024.jpeg', 0)



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
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/002_1024.jpeg')

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
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/coronavirus-en-chine.jpg')

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
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/0109-00176-096b1.jpg')

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
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/CHINA-HEALTH-VIRUS-134752 (1).jpg')

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
image = cv2.imread('/kaggle/input/medical-masks-dataset/images/nouveau-virus-en-chine-la-ville-de-wuhan-mise-en-quarantaine-1.jpg')

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
image = cv2.imread('/kaggle/input/medical-masks-dataset/images//organizacion-preocupacion-potencial-virus-propague.jpg')

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

image = cv2.imread('/kaggle/input/medical-masks-dataset/images/so(19).jpg')

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
import numpy as np

import pandas as pd 

import cv2

from fastai.vision import *

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

import os

import shutil

from glob import glob

%matplotlib inline

!pip freeze > '../working/dockerimage_snapshot.txt'
def makeWordCloud(df,column,numWords):

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=numWords,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def plotImages(artist,directory):

    print(artist)

    multipleImages = glob(directory)

    plt.rcParams['figure.figsize'] = (15, 15)

    plt.subplots_adjust(wspace=0, hspace=0)

    i_ = 0

    for l in multipleImages[:25]:

        im = cv2.imread(l)

        im = cv2.resize(im, (128, 128)) 

        plt.subplot(5, 5, i_+1) #.set_title(l)

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

        i_ += 1



np.random.seed(7)
print(os.listdir("../input/medical-masks-dataset/images/"))
img_dir='../input/medical-masks-dataset/images'

path=Path(img_dir)

data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3),

                                  size=299,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=8, figsize=(40,40))