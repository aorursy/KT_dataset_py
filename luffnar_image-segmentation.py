import cv2 as cv

import numpy as np

import matplotlib.pyplot as plt

from scipy import ndimage

from skimage.color import rgb2gray

image = plt.imread("../input/1.jpeg")

image.shape

plt.imshow(image)
gray = rgb2gray(image)

plt.imshow(gray, cmap = 'gray')
gray.shape
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])

# mean = gray_r.mean()

# print(mean)

for i in range(gray_r.shape[0]):

    if gray_r[i] > gray_r.mean():

        gray_r[i] = 1

    else:

        gray_r[i] = 0

gray = gray_r.reshape(gray.shape[0],gray.shape[1])

ret,thresh1 = cv.threshold(image,127,255,cv.THRESH_BINARY)

ret,thresh2 = cv.threshold(image,127,255,cv.THRESH_BINARY_INV)

ret,thresh3 = cv.threshold(image,127,255,cv.THRESH_TRUNC)

ret,thresh4 = cv.threshold(image,127,255,cv.THRESH_TOZERO)

ret,thresh5 = cv.threshold(image,127,255,cv.THRESH_TOZERO_INV)



titles = ['Mean','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]



for i in range(6):

    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')

    plt.title(titles[i])

    plt.xticks([]),plt.yticks([])

plt.show()
gray = rgb2gray(image)

gray_r = gray.reshape(gray.shape[0]*gray.shape[1])

for i in range(gray_r.shape[0]):

    if gray_r[i] > gray_r.mean():

        gray_r[i] = 3

    elif gray_r[i] > 0.5:

        gray_r[i] = 2

    elif gray_r[i] > 0.25:

        gray_r[i] = 1

    else:

        gray_r[i] = 0

gray_l = gray_r.reshape(gray.shape[0],gray.shape[1])

src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ret,th1 = cv.threshold(image,127,255,cv.THRESH_BINARY)

th2 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_MEAN_C,\

            cv.THRESH_BINARY,11,2)

th3 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\

            cv.THRESH_BINARY,11,2)

titles = ['Local Method', 'Global Thresholding (v = 127)',

            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

images = [gray_l, th1, th2, th3]

for i in range(4):

    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')

    plt.title(titles[i])

    plt.xticks([]),plt.yticks([])

plt.show()
#same image - 

img = (plt.imread("../input/1.jpeg"))/255 #dividing by 255 to bring the pixel values between 0 and 1

print(img.shape)

# 3 is the number of channels
'''Covert the image into 2-D array'''

img1 = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

img1.shape
'''Fit the k-means algo on our converted image'''

from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=4, random_state=0).fit(img1)

img2 = kmeans.cluster_centers_[kmeans.labels_]
cluster_img = img2.reshape(img.shape[0], img.shape[1], img.shape[2])

plt.imshow(cluster_img)
img = cv.imread('../input/1.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 0, 255, 

                            cv.THRESH_BINARY_INV +

                            cv.THRESH_OTSU) 

plt.imshow(thresh, 'gray')
# Noise removal using Morphological 

# closing operation 

kernel = np.ones((5, 5), np.uint8) 

closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 2) 

opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)

# Background area using Erosion 

bg1 = cv.erode(closing, kernel, iterations = 1) 



#Background using dilation

bg2 = cv.dilate(closing, kernel, iterations = 1)



#Background using opening

bg3 = cv.erode(opening, kernel, iterations = 1) 

bg4 = cv.dilate(opening, kernel, iterations = 1) 

titles = ['ErosionClose', 'DilationCLose', 'ErosionOpen', 'DilationOpen']

images = [bg1, bg2, bg3, bg4]

for i in range(4):

    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')

    plt.title(titles[i])

    plt.xticks([]),plt.yticks([])

plt.show()
img = cv.imread('../input/1.jpeg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal

kernel = np.ones((5,5),np.uint8)

closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 2) 

# sure background area

sure_bg = cv.dilate(closing,kernel,iterations=3)

# Finding sure foreground area

dist_transform = cv.distanceTransform(closing,cv.DIST_L2,5)

ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region

sure_fg = np.uint8(sure_fg)

unknown = cv.subtract(sure_bg,sure_fg)



# Marker labelling

ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1

markers = markers+1

# Now, mark the region of unknown with zero

markers[unknown==255] = 0



markers = cv.watershed(img,markers)

img[markers == -1] = [255,0,0]

#plt.imshow(markers)

plt.imshow(img)