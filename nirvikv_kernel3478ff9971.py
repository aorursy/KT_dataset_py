import numpy as np

import cv2

#from google.colab.patches import cv2_imshow

import matplotlib.pyplot as plt

%matplotlib inline



from scipy.ndimage import gaussian_filter

from skimage import data

from skimage import img_as_float

from skimage.morphology import reconstruction

import skimage.draw

import time

import glob

import os



images_path = '../input/lemons/'

# construct the argument parser and parse the arguments
image = cv2.imread('../input/lemon6.jpg')

plt.figure(figsize=(16,10)) # to set the figure size

plt.imshow(image)

plt.show()


image = cv2.imread('../input/lemon6.jpg')

  #output = image.copy()

  # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # cv2_imshow(hsv)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

i = image

#plt converts images to 3dim

plt.imshow(image, cmap = 'gray')




# Convert to float: Important for subtraction later which won't work with uint8

image = gaussian_filter(image, 1)



seed = np.copy(image)

seed[:, :-1] = image.min()

mask = image



dilated = reconstruction(seed, mask, method='dilation')
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,

                                    ncols=3,

                                    figsize=(8, 2.5),

                                    sharex=True,

                                    sharey=True)



ax0.imshow(image, cmap='gray')

ax0.set_title('original image')

ax0.axis('off')



ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')

ax1.set_title('dilated')

ax1.axis('off')



ax2.imshow(image - dilated, cmap='gray')

ax2.set_title('image - dilated')

ax2.axis('off')



fig.tight_layout()
img = (dilated-i).astype(np.uint8)

# print(img.shape)

# print(image.shape)

rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # print(rgb.shape)

hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2XYZ)

plt.imshow(hsv)
from skimage.filters import threshold_multiotsu



thresholds = threshold_multiotsu(hsv)

regions = np.digitize(image, bins=thresholds)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))



# Plotting the original image.

ax[0].imshow(image, cmap='gray')

ax[0].set_title('Original')

ax[0].axis('off')



# Plotting the histogram and the two thresholds obtained from

# multi-Otsu.

ax[1].hist(image.ravel(), bins=255)

ax[1].set_title('Histogram')

for thresh in thresholds:

    ax[1].axvline(thresh, color='r')



# Plotting the Multi Otsu result.

ax[2].imshow(regions, cmap='Accent')

ax[2].set_title('Multi-Otsu result')

ax[2].axis('off')



plt.subplots_adjust()



plt.show()
# def inverte(imagem):

#     imagem = (255-imagem)

#     return imagem
output = image.copy()

rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

img = cv2.blur(img,(5,5))



img = img + 100

plt.imshow(img)

# new_image = np.zeros(img.shape, img.dtype)

# for y in range(img.shape[0]):

#     for x in range(img.shape[1]):

#           new_image[y,x] = np.clip(2*img[y,x] + 10, 0, 255)



# cv2_imshow(new_image)

#invert_img = inverte(img)

#cv2_imshow(invert_img)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 125, 150, 150, 10, 10)

# ensure at least some circles were found

if circles is not None:

  # convert the (x, y) coordinates and radius of the circles to integers

  circles = np.round(circles[0, :]).astype("int")

  # loop over the (x, y) coordinates and radius of the circles

  mean_r = int(np.mean(circles, axis = 0)[2])

  # print(mean_r)

  for (x, y, r) in circles:

    # draw the circle in the output image, then draw a rectangle

    # corresponding to the center of the circle

    if(r < mean_r):

      cv2.circle(output, (x, y), r, (0, 255, 0), 4)

    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

  # show the output image

  plt.imshow(np.hstack([image, output]))


def load_bnw_image(name):

    image = cv2.imread(name)

    #output = image.copy()

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # cv2_imshow(hsv)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image
def generate_dilated(image):

    image = gaussian_filter(image, 1)



    seed = np.copy(image)

    seed[:, :-1] = image.min()

    mask = image



    dilated = reconstruction(seed, mask, method='dilation')

    return dilated
def generate_circles(gray, acc=3, min_Dist=[125]):

    for m in min_Dist:

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, acc, m, 120) #, 150, 10, 10)

    return circles
def draw_circles(circles, image):

    output = image.copy()

    if circles is not None:

        # convert the (x, y) coordinates and radius of the circles to integers

        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles

        mean_r = int(np.mean(circles, axis = 0)[2])

        # print(mean_r)

        for (x, y, r) in circles:

          # draw the circle in the output image, then draw a rectangle

          # corresponding to the center of the circle

            if(r < mean_r):

                cv2.circle(output, (x, y), r, (0, 255, 0), 4)

            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)

    return output
def split_squares(img):

    '''h = img.shape[0]

                h = int(h/2)

                return [img[:h, :h, :], img[h:2*h, :h, :], img[:h, h:2*h, :], img[h:2*h, h:2*h, :]]'''

    print(img.shape, 'img')



    window_r = 150

    window_c = 150

    l = []

    for r in range(0, img.shape[0]-window_r+1, window_r):

        for c in range(0, img.shape[1]-window_c+1, window_c):

            window = img[r:r+window_r,c:c+window_c]

            l.append(window)

  

    return l


def generate_masks(circles, img):

    if circles is not None:

        # convert the (x, y) coordinates and radius of the circles to integers

        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles

        mean_r = int(np.mean(circles, axis = 0)[2])

        # print(mean_r)

        op_list = []

        for i, (x, y, r) in enumerate(circles):

            output = np.zeros((img.shape))

          # draw the circle in the output image, then draw a rectangle

          # corresponding to the center of the circle

            if(r < mean_r):

                cv2.circle(output, (x, y), r, (255, 255, 255), -1)

            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)

            op_list.append(output)

            

    l = len(op_list)

    fig = plt.figure()

    r = 5

    c = 5

    for i, img in enumerate(op_list):

        aux = fig.add_subplot(r, c, i+1)

        imgplot = plt.imshow(img)

        aux.set_title("index: {}".format(i))

        

    plt.axis("off")

    plt.show()
img_wp = load_bnw_image('../input/lemon6.jpg')

start = time.time()

circles_wp = generate_circles(img_wp)

end = time.time()

output_wp = draw_circles(circles_wp, img_wp)

plt.imshow(np.hstack([img_wp, output_wp]))

print("Number of Lemons: ", len(circles_wp[0]))

print('time taken: ', (end-start))
import time

def main(name):

  #load image

    start = time.time()

    img_fs = load_bnw_image(name)

    img2_fs = img_fs.copy()



    #Custom edge detection

    dilated_fs = generate_dilated(img_fs)

    img_fs = (img2_fs-dilated_fs).astype(np.uint8)



    #convert to XYZ

    rgb_fs = cv2.cvtColor(img_fs, cv2.COLOR_GRAY2BGR)

    hsv_fs = cv2.cvtColor(rgb_fs, cv2.COLOR_BGR2XYZ)

    # cv2_imshow(hsv)



    #map XYZ as HSV and Convert to Gray for HoughCircles

    rgb_fs = cv2.cvtColor(hsv_fs, cv2.COLOR_HSV2BGR)

    img_fs = cv2.cvtColor(rgb_fs, cv2.COLOR_BGR2GRAY)

    img_fs = cv2.blur(img_fs,(5,5))

    print("Input image")

    plt.imshow(img_fs)

    circles_fs = generate_circles(img_fs)

    output_fs = draw_circles(circles_fs, img2_fs)



    generate_masks(circles_fs, img2_fs)

    end = time.time()



    plt.imshow(np.hstack([img2_fs, output_fs]))

    print('Number of Lemons: ', len(circles_fs[0]))

    print('time taken: ', (end-start))
main('../input/lemon6.jpg')
# import cv2

# import numpy as np

# from google.colab.patches import cv2_imshow



# output = cv2.imread('lemons1.jpeg')

# orig = cv2.imread('lemons1.jpeg')

# height = orig.shape[0]

# width = orig.shape[1]

# # create tmp images

# rrr = np.zeros((height,width,1), np.uint8)

# rrr[:,: ,:] = 255

# ggg = rrr.copy()

# bbb = rrr.copy()

# processed = rrr.copy()

# storage = np.zeros((height,width,1), np.uint8)



# def channel_processing(channel):

#     pass

#     channel = cv2.adaptiveThreshold(channel, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=55, C=7)

#     #mop up the dirt

#     channel = cv2.dilate(channel,(3, 3),iterations = 1)

#     channel = cv2.erode(channel,(3, 3),iterations = 1)



# def inter_centre_distance(x1,y1,x2,y2):

#     return ((x1-x2)**2 + (y1-y2)**2)**0.5



# def colliding_circles(circles):

#     for index1, circle1 in enumerate(circles):

#         for circle2 in circles[index1+1:]:

#             x1, y1, Radius1 = circle1[0]

#             x2, y2, Radius2 = circle2[0]

#             #collision or containment:

#             if inter_centre_distance(x1,y1,x2,y2) < Radius1 + Radius2:

#                 return True



# def find_circles(processed, storage, LOW):

#     try:

#         cv2.HoughCircles(processed, storage, cv.CV_HOUGH_GRADIENT, 2, 32.0, 30, LOW)#, 0, 100) great to add circle constraint sizes.

#     except:

#         LOW += 1

#         print ('try')

#         find_circles(processed, storage, LOW)

#     circles = np.asarray(storage)

#     print ('number of circles:', len(circles))

#     if colliding_circles(circles):

#         LOW += 1

#         storage = find_circles(processed, storage, LOW)

#     print ('c', LOW)

#     return storage



# def draw_circles(storage, output):

#     circles = np.asarray(storage)

#     print (len(circles), 'circles found')

#     for circle in circles:

#         Radius, x, y = int(circle[0][2]), int(circle[0][0]), int(circle[0][1])

#         cv2.Circle(output, (x, y), 1, cv.CV_RGB(0, 255, 0), -1, 8, 0)

#         cv.Circle(output, (x, y), Radius, cv.CV_RGB(255, 0, 0), 3, 8, 0)



# #split image into RGB components

# for i in [0, 1, 2]:

#     colour = orig.copy()

#     if i != 0: colour[:,:,0] = 0

#     if i != 1: colour[:,:,1] = 0

#     if i != 2: colour[:,:,2] = 0



#     if i == 0: bbb = colour[:, :, 0]

#     if i == 1: ggg = colour[:, :, 1]

#     if i == 2: rrr = colour[:, :, 2]



# cv2_imshow(np.vstack([bbb, ggg, rrr]))

# #process each component

# channel_processing(rrr)

# channel_processing(ggg)

# channel_processing(bbb)

# #combine images using logical 'And' to avoid saturation

# #rrr = cv2.bitwise_and(rrr, bbb)

# processed = cv2.bitwise_and(rrr, ggg)

# # cv.SaveImage('case3_processed.jpg',processed)

# #use canny, as HoughCircles seems to prefer ring like circles to filled ones.

# processed_dilated = generate_dilated(processed)

# cv2_imshow((processed-processed_dilated))



# #smooth to reduce noise a bit more

# cv.Smooth(processed, processed, cv.CV_GAUSSIAN, 7, 7)

# cv_imshow('processed', processed)

# #find circles, with parameter search

# storage = find_circles(processed, storage, 100)

# draw_circles(storage, output)

# # show images

# cv_imshow("original with circles", output)

# # # cv.SaveImage('case1.jpg',output)



# # cv.WaitKey(0)

from skimage import io, filters

from scipy import ndimage

import matplotlib.pyplot as plt



im = io.imread('../input/lemon6.jpg', as_gray=True)

val = 0.2

drops = ndimage.binary_fill_holes(im < val)

plt.imshow(drops, cmap='gray')

plt.show()

import cv2

import numpy as np

img = cv2.imread('../input/lemon6.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dilated = generate_dilated(gray)

img = (gray-dilated).astype(np.uint8)

contours,h = cv2.findContours(img,1,2)

for cnt in contours:

    cv2.drawContours(img,[cnt],0,(0,0,255),1)

    

plt.imshow(img)
