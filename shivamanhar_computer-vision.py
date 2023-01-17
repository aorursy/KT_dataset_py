import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

from PIL import Image



'''import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))'''



# Any results you write to the current directory are saved as output.
img_green = cv2.imread("/kaggle/input/open-images-2019-object-detection/test/34ff94b34c6851bf.jpg")

print(img_green.shape)

plt.imshow(img_green)

plt.show()
img = cv2.imread("/kaggle/input/open-images-2019-object-detection/test/149d7a017153bc72.jpg")

plt.imshow(img)

plt.show()
print(img.shape)
# Convert the image into RGB

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img_rgb.shape)

plt.imshow(img_rgb)

plt.show()
bicycle1 = np.copy(img_rgb)

bicycle1[:,:,0] = 5*bicycle1[:,:,0]

bicycle1[:,:,1] = bicycle1[:,:,1]/2 
plt.imshow(bicycle1)

plt.show()
bicycle2 = np.copy(img_rgb)

cv2.rectangle(bicycle2, pt1=(100,400), pt2=(200, 600), color=(0, 255,0), thickness=5)

plt.imshow(bicycle2)

plt.show()
bicycle3 = np.copy(img_rgb)

cv2.circle(bicycle3, center=(200, 200), radius=50, thickness=5, color=(0, 0, 255))

plt.imshow(bicycle3)

plt.show()
bicycle4 = img_rgb.copy()

cv2.putText(bicycle4, text="Sense Tech", org=(250, 260), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 10, 100), thickness=2, lineType=cv2.LINE_AA )

plt.imshow(bicycle4)

plt.show()
fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

ax1.hist(bicycle4[:,:,0].ravel(), bins=20)

ax2.hist(bicycle4[:,:,1].ravel(), bins=20)

ax3.hist(bicycle4[:,:,2].ravel(), bins=20)
'''bicycle5 = np.copy(img_rgb)

def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        cv2.circle(bicycle5, center = (x,y), radius=5, color=(87, 184, 237), thinkness=-1)

    elif event == cv2.EVENT_RBUTTONDOWN:

        cv2.circle(bicycle5, center=(x,y), radius=10, color=(85, 185, 234), thinkness=1)



cv2.namedWindow(winname='my_drawing')

cv2.setMouseCallback('my_drawing', draw_circle)



while True:

    cv2.imshow('my_drawing', bicycle5)

    if cv2.waitKey(10) & 0xFF == 27:

        break

    cv2.destroyAllWindows()'''
#Average blurring

bicycle5 = np.copy(img_rgb)

#

kernels = [5, 11, 17]



plt.imshow(bicycle5)

plt.show()

fig, axs = plt.subplots(nrows= 1, ncols=3, figsize=(20,20))

for ind, s in enumerate(kernels):

    img_blurred = cv2.blur(bicycle5, ksize=(s,s))

    ax = axs[ind]

    ax.imshow(img_blurred)

    ax.axis('off')

plt.show()
img_0 = cv2.blur(img_rgb, ksize=(7,7))

img_1 = cv2.GaussianBlur(img_rgb, ksize=(7,7), sigmaX=0)

img_2 = cv2.medianBlur(img_rgb, 7)

img_3 = cv2.bilateralFilter(img_rgb, 7, sigmaSpace=75, sigmaColor=75)



images = [img_0, img_1, img_2, img_3]

fig, axs = plt.subplots(nrows =1, ncols =4, figsize=(20,20))



for ind, p in enumerate(images):

    ax = axs[ind]

    ax.imshow(p)

    ax.axis('off')

    

plt.show()
_, thresh_0 = cv2.threshold(img_rgb, 127, 255, cv2.THRESH_BINARY)

_, thresh_1 = cv2.threshold(img_rgb, 127, 255, cv2.THRESH_BINARY_INV)

_, thresh_2 = cv2.threshold(img_rgb, 127, 255, cv2.THRESH_TOZERO)

_, thresh_3 = cv2.threshold(img_rgb, 127, 255, cv2.THRESH_TOZERO_INV)

_, thresh_4 = cv2.threshold(img_rgb, 127, 255, cv2.THRESH_TRUNC)



images = [img, thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]



fig, axs = plt.subplots(nrows= 2, ncols=3, figsize=(13,13))



for ind, p in enumerate(images):

    ax = axs[ind//3, ind%3]

    ax.imshow(p)

    

plt.show()
sobel_x = cv2.Sobel(img_rgb, cv2.CV_64F, dx=1, dy=0, ksize=5)

sobel_y = cv2.Sobel(img_rgb, cv2.CV_64F, dx=0, dy=1, ksize=5)



blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2 = sobel_y, beta=0.5, gamma=0)

laplacian = cv2.Laplacian(img_rgb, cv2.CV_64F)
images = [sobel_x, sobel_y, blended, laplacian]



plt.figure(figsize=(20,20))

for i in range(4):

    plt.subplot(1, 4, i+1)

    plt.imshow(images[i], cmap='gray')

    plt.axis('off')

    

plt.show()
index_210 = img_rgb > 110

bicycle6 = np.copy(img_rgb)
bicycle6[index_210] = 250
plt.figure(figsize=(16,6))

plt.imshow(bicycle6)

plt.show()
small_img = np.copy(img_rgb)

small_img = small_img[350:-200:]

print(small_img.shape)

plt.figure(figsize=(20,20))

plt.imshow(small_img)

plt.axis('off')

plt.show()

edges = cv2.Canny(image= small_img, threshold1=250, threshold2=250)

edges[100:,:,]

plt.figure(figsize=(20,20))

plt.imshow(edges)

plt.axis('off')

plt.show()

#Set the lower and upper threshold

med_val = np.median(small_img)

lower = int(max(0, .7*med_val))

upper = int(min(255, 1.3*med_val))

# Blurring with ksize= 5

img_k5 = cv2.blur(small_img, ksize=(5,5))



#Canny detection with different thresholds

edges_k5 = cv2.Canny(img_k5, threshold1=lower, threshold2 = upper)



edges_k5_2 = cv2.Canny(img_k5, lower, upper+100)



#blurring with ksize=9

img_k9 = cv2.blur(small_img, ksize=(9,9))



#Canny detection with different thresholds

edges_k9 = cv2.Canny(img_k9, lower, upper)

edges_k5_2 = cv2.Canny(img_k9, lower, upper)



#plot the images

images = [edges_k5, edges_k5, edges_k9, edges_k5_2]

plt.figure(figsize=(20, 5))



for i in range(4):

    plt.subplot(2,2, i+1)

    plt.imshow(images[i])

    plt.axis('off')

plt.show()





arr = np.array([[[3, 4], [8, 2], [1, 9]],

               [[3, 0], [8, 2], [1, 10]],

               [[3, 4], [8, 2], [1, 9]]])

arr.shape
plt.imshow(arr[:,:,1],  cmap='gray')

plt.show()
arr[:,:,1]
arr = np.array([[[3, 4, 5], [8, 2, 5], [1, 9, 5]],

               [[3, 0, 1], [8, 2, 7], [1, 10, 3]],

               [[3, 44, 6], [8, 2, 7], [1, 9, 8]],

               [[3, 4, 0], [8, 2, 1], [1, 9,1 ]],

               [[3, 0, 1], [98, 26, 88], [1, 10,1]],

               [[250, 250,250], [120, 110, 250], [1, 9, 250]],

               [[3, 4, 4], [8, 2,6], [1, 9, 4]],

               [[3, 0, 2], [8, 2, 5], [1, 10,2]],

               [[3, 4, 0], [8, 2,1], [1, 9, 3]]])

arr.shape
arr2 = np.copy(arr)
plt.imshow(arr)

plt.show()
arr_condition = arr < 250

arr2[arr_condition] += 100 
plt.imshow(arr2)

plt.show()
# Convert the image into gray scale

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img_gray.shape)

plt.figure(figsize=(16, 8))

plt.imshow(img_gray, cmap = 'gray')

plt.show()
np_img = np.array(img_gray)
print(np_img[10:-5, 10:-5])
img2 = np.copy(img_gray)

img2[100:200, 10:100] = 0
plt.imshow(img2, cmap = 'gray')

plt.show()
img_green = cv2.cvtColor(img_green, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 8))

plt.imshow(img_green)

plt.show()
img_green = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)
img3 = np.copy(img2)

img3[100:200, 10:100] = img_green[100:200, 10:100] 

plt.figure(figsize=(16, 8))

plt.imshow(img3, cmap = 'gray')

plt.show()