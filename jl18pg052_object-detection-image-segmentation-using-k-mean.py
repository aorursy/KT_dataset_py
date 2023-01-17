# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2, os

import numpy as np

import seaborn as sns

import urllib.request as lib

import requests

import matplotlib.pyplot as plt
cv2.__version__
#image = lib.urlopen('http://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png')

!wget https://ae01.alicdn.com/kf/HTB1TVk9SFXXXXabXXXXq6xXFXXX3/Fashion-Wristwatch-New-Wrist-Watch-Men-Watches-Top-Brand-Luxury-Famous-Quartz-Watch-for-Men-Male.jpg
os.listdir()
img = cv2.imread("Fashion-Wristwatch-New-Wrist-Watch-Men-Watches-Top-Brand-Luxury-Famous-Quartz-Watch-for-Men-Male.jpg",1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (12,8))

plt.title("Image of a wrist-watch", size = 20)

plt.imshow(img)

plt.xticks([]), plt.yticks([])
sns.distplot(img.flatten(),kde = False)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

ret,thresh1 = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)



plt.figure(figsize = (12,8))

plt.imshow(thresh1)
img.shape
sobelx = cv2.Sobel(img,int(cv2.CV_64F),1,0,ksize=3) #ksize=3 means we'll be using the 3x3 Sobel filter

sobely = cv2.Sobel(img,int(cv2.CV_64F),0,1,ksize=3)



#To plot the vertical and horizontal edge detectors side by side

plt.figure(figsize=(14,16))

plt.subplot(2,2,1)

plt.imshow(sobelx, cmap = 'gray')

plt.title('Sobel X (vertical edges)')

plt.xticks([])

plt.yticks([])



plt.subplot(2,2,2)

plt.imshow(sobely, cmap = 'gray')

plt.xticks([])

plt.yticks([])

plt.title('Sobel Y (horizontal edges)')
height, width = img.shape[0:2]
startRow = int(height*.12)

startCol = int(width*.12)

endRow = int(height*.90)

endCol = int(width*.90)
cropped_image = img[startRow:endRow, startCol:endCol]
plt.figure(figsize = (12,18))



plt.subplot(1,2,1)

plt.imshow(img)

plt.title("Original Image", size = 20)

plt.xticks([]), plt.yticks([])



plt.subplot(1,2,2)

plt.imshow(cropped_image)

plt.title("Cropped Image", size = 20)

plt.xticks([]), plt.yticks([])
newImg = cv2.resize(img, (500, 750))

plt.figure(figsize = (12,8))

plt.imshow(newImg)
plt.figure(figsize = (14,16))



plt.subplot(1,3,1)

plt.title("Original Image", size = 20)

plt.imshow(img)

plt.xticks([]),plt.yticks([])



plt.subplot(1,3,2)

contrast_img = cv2.addWeighted(img, 3, np.zeros(img.shape, img.dtype), 0, 0)

plt.title("Contrasted Image", size = 20)

plt.imshow(contrast_img)

plt.xticks([]),plt.yticks([])



plt.subplot(1,3,3)

contrast_img = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0, 127)

plt.title("Image with high brightness & 0 contrast", size = 15)

plt.imshow(contrast_img)

plt.xticks([]),plt.yticks([])
cont_gray = cv2.cvtColor(contrast_img, cv2.COLOR_RGB2GRAY)

equ = cv2.equalizeHist(cont_gray)

improved = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5,5))

cl1 = clahe.apply(cont_gray)



plt.figure(figsize = (14,20))



plt.subplot(1,3,1)

plt.imshow(cont_gray)

plt.title("Original Gray Image",size = 16)

plt.xticks([]), plt.yticks([])



plt.subplot(1,3,2)

plt.imshow(equ)

plt.title("After equalizing", size = 16)

plt.xticks([]), plt.yticks([])



plt.subplot(1,3,3)

plt.imshow(cl1)

plt.title("After equalizing using CLAHE", size = 16)

plt.xticks([]), plt.yticks([])
plt.figure(figsize = (16,5))



plt.subplot(1,3,1)

sns.distplot(contrast_img.flatten(), kde = False)

plt.title("Histogram of contrasted Image", size = 16)



plt.subplot(1,3,2)

sns.distplot(equ.flatten(), kde = False, color = "lime")

plt.title("Histogram after equalizing", size = 16)



plt.subplot(1,3,3)

sns.distplot(cl1.flatten(), kde = False, color = "crimson")

plt.title("Equalizing using CLAHE", size = 16)
blur_img_1 = cv2.GaussianBlur(img, (7,7), cv2.BORDER_REFLECT_101)

blur_img_2 = cv2.GaussianBlur(img, (11,11), cv2.BORDER_REFLECT_101)

plt.figure(figsize = (14,18))



plt.subplot(1,2,1)

plt.title("Gausian Blurred image", size = 20)

plt.imshow(blur_img_1)

plt.xticks([]), plt.yticks([])



plt.subplot(1,2,2)

plt.title("Gausian Blurred with high kernel size", size = 20)

plt.imshow(blur_img_2)

plt.xticks([]), plt.yticks([])
edge = cv2.Laplacian(img, -1, ksize = 5, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)

plt.figure(figsize = (12,8))

plt.title("Laplacian High Pass Filter", size = 20)

plt.imshow(edge)

plt.xticks([]), plt.yticks([])
sns.distplot(edge.flatten(), kde = False)
med_blur_image = cv2.medianBlur(img,5)

plt.figure(figsize = (12,8))

plt.title("Blurring using Median technique", size = 20)

plt.imshow(med_blur_image)
plt.figure(figsize = (18, 24))



plt.subplot(1,3,1)

edge_img = cv2.Canny(img,100,200)

plt.imshow(edge_img)

plt.title("Edge detection", size = 20)

plt.xticks([]), plt.yticks([])



plt.subplot(1,3,2)

plt.imshow(img)

plt.title("Original Image", size = 20)

plt.xticks([]), plt.yticks([])



plt.subplot(1,3,3)

edge_img_2 = cv2.Canny(img,50,100)

plt.imshow(edge_img_2)

plt.title("Gradient thresh changed", size = 14)

plt.xticks([]), plt.yticks([])
!wget https://thumbs.dreamstime.com/z/adult-man-wearing-jeans-plaid-shirt-red-cap-isolated-white-background-36005651.jpg
os.listdir()
img2 = cv2.imread("adult-man-wearing-jeans-plaid-shirt-red-cap-isolated-white-background-36005651.jpg",-1)

img_rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
plt.figure(figsize = (16,22))



plt.subplot(1,3,1)

plt.title("Image in RGB format", size = 20)

plt.imshow(img_rgb)

plt.xticks([]),plt.yticks([])



plt.subplot(1,3,2)

plt.title("Image in BGR format", size = 20)

plt.imshow(img2)

plt.xticks([]),plt.yticks([])



plt.subplot(1,3,3)

img_rgb2hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

plt.title("Image in HSV format", size = 20)

plt.imshow(img_rgb2hsv)

plt.xticks([]), plt.yticks([])
lower_red = np.array([0, 100, 100])#,np.uint8

upper_red = np.array([5, 255, 255])#,np.uint8



hsv_img = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)



image_thresh = cv2.inRange(hsv_img, lower_red, upper_red)

res = cv2.bitwise_and(hsv_img, hsv_img, mask= image_thresh)

#cv2.imwrite('output2.jpg', frame_threshed)

plt.figure(figsize = (12,12))

plt.title("Color Detection", size = 20)

plt.imshow(res)

plt.xticks([]), plt.yticks([])
!wget https://i.ytimg.com/vi/B7xrGE8GHD4/maxresdefault.jpg
os.listdir()
img3 = cv2.imread("maxresdefault.jpg", -1)

img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (12,12))

plt.imshow(img3_rgb)

plt.xticks([]), plt.yticks([])
img3_rgb.shape
pixel_values = img3_rgb.reshape((-1, 3))

pixel_values.shape
df = pd.DataFrame(pixel_values)

df.head()
pixel_values = np.float32(pixel_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.2)
k = 6

_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convert back to 8 bit values

centers = np.uint8(centers)



# flatten the labels array

labels = labels.flatten()
segmented_image = centers[labels.flatten()]
# reshape back to the original image dimension

segmented_image = segmented_image.reshape(img3_rgb.shape)

plt.figure(figsize = (12,12))

plt.imshow(segmented_image)

plt.xticks([]), plt.yticks([])
#f, axarr = plt.subplots(1,2, figsize = (22,12))

#axarr[0].imshow(img3_rgb)

#axarr[1].imshow(segmented_image)



plt.figure(figsize = (22,12))

plt.subplot(1, 2, 1)

plt.imshow(img3_rgb)

plt.title("Original Image", size = 16)

plt.xticks([]), plt.yticks([])



plt.subplot(1, 2, 2)

plt.imshow(segmented_image)

plt.title("Segmented Image", size = 16)

plt.xticks([]), plt.yticks([])
!wget https://cdn.abcotvs.com/dip/images/4041625_082318-kgo-sky7-hazy-skies-vid.jpg
os.listdir()
img_pollu = cv2.imread("4041625_082318-kgo-sky7-hazy-skies-vid.jpg", -1)

pollu_rgb = cv2.cvtColor(img_pollu, cv2.COLOR_BGR2RGB)



plt.figure(figsize = (12,8))

plt.imshow(pollu_rgb)

plt.title("Original Image", size = 20)

plt.xticks([]), plt.yticks([])
sns.distplot(pollu_rgb.flatten(), kde = False)
pollu_gray = cv2.cvtColor(pollu_rgb,cv2.COLOR_RGB2GRAY)

equ = cv2.equalizeHist(pollu_gray)



plt.figure(figsize = (18,26))

plt.subplot(1,3,1)

plt.imshow(pollu_gray)

plt.title("Gray Scale Original Image", size = 18)

plt.xticks([]), plt.yticks([])



plt.subplot(1,3,2)

plt.imshow(equ)

plt.title("Gray Scale Image after equalizing", size = 18)

plt.xticks([]), plt.yticks([])



plt.subplot(1,3,3)

clahe_1 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,5))

cl2 = clahe_1.apply(pollu_gray)

plt.imshow(cl2)

plt.title("Gray Scale Image using CLAHE", size = 18)

plt.xticks([]), plt.yticks([])
plt.figure(figsize = (18,20))



plt.subplot(1,2,1)

plt.imshow(pollu_rgb)

plt.title("Original Image", size = 20)

plt.xticks([]), plt.yticks([])



plt.subplot(1,2,2)

pollu_con_rgb = cv2.cvtColor(cl2, cv2.COLOR_GRAY2RGB)

plt.imshow(pollu_con_rgb)

plt.title("Image after CLAHE & after converting to RGB", size = 16)

plt.xticks([]),plt.yticks([])
plt.figure(figsize = (16,5))



plt.subplot(1,3,1)

sns.distplot(pollu_gray.flatten(), kde = False, color = "red")

plt.title("Pixels Distribution for gray_scale", size = 14)



plt.subplot(1,3,2)

sns.distplot(equ.flatten(), kde = False, color = "green")

plt.title("Pixels Distribution after histogram equalization", size = 12)



plt.subplot(1,3,3)

sns.distplot(cl2.flatten(), kde = False, color = "orange")

plt.title("Pixels Distribution after CLAHE", size = 12)