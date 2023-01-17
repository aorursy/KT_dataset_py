import cv2

import matplotlib.pyplot as plt



def plt_imshow(img, is_color=1, continue_plotting=0):

    if is_color:

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([])

    else:

        plt.imshow(img, cmap="gray")

        

    if continue_plotting==0:

        plt.show()

    

import numpy as np
# load the image

image = cv2.imread("../input/Enlight90.jpg")

plt_imshow(image)

# compute the ratio of the new height

# to the old height, clone it, and resize it



height, width, _ = image.shape

angle            = 180

scale            = 1.0

M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)

image = cv2.warpAffine(image, M, (width, height)) 



ratio = 500.0/image.shape[0] 

orig  = image.copy()

image = cv2.resize(image, (0,0), fy=ratio, fx=ratio)



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(gray, 75, 200)

 

# show the original image and the edge detected image

print("STEP 1: Edge Detection")

plt_imshow(image)

plt_imshow(edged)

# find the contours in the edged image, keeping only the

# ones with largest area

cnts,_   = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts       = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]



# loop over the contours

for c in cnts:

    # approximate the contour

    peri   = cv2.arcLength(c, True)

    accuracy_of_approximation = 0.02 * peri

    approx = cv2.approxPolyDP(c, accuracy_of_approximation, True) #  Douglas-Peucker algorithm

    

    # if the approximated contour has four points, then we

    # can assume that we have found our screen

    if len(approx) == 4:

        screenCnt = approx

        break



#  show the contour (outline) of the piece of paper

print("STEP 2: Find contours of paper")

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

plt_imshow(image)
def order_points(pts):

    # initialzie a list of coordinates that will be ordered

    # such that the first entry in the list is the top-left,

    # the second entry is the top-right, the third is the

    # bottom-right, and the fourth is the bottom-left

    rect = np.zeros((4, 2), dtype = "float32")



    # the top-left point will have the smallest sum, whereas

    # the bottom-right point will have the largest sum

    s = pts.sum(axis = 1)

    rect[0] = pts[np.argmin(s)]

    rect[2] = pts[np.argmax(s)]



    # now, compute the difference between the points, the

    # top-right point will have the smallest difference,

    # whereas the bottom-left will have the largest difference

    diff = np.diff(pts, axis = 1)

    rect[1] = pts[np.argmin(diff)]

    rect[3] = pts[np.argmax(diff)]



    # return the ordered coordinates

    return rect
def four_point_transform(image, pts):

    # obtain a consistent order of the points and unpack them

    # individually

    rect = order_points(pts)

    (tl, tr, br, bl) = rect



    # compute the width of the new image, which will be the

    # maximum distance between bottom-right and bottom-left

    # x-coordiates or the top-right and top-left x-coordinates

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))



    # compute the height of the new image, which will be the

    # maximum distance between the top-right and bottom-right

    # y-coordinates or the top-left and bottom-left y-coordinates

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxHeight = max(int(heightA), int(heightB))



    # now that we have the dimensions of the new image, construct

    # the set of destination points to obtain a "birds eye view",

    # (i.e. top-down view) of the image, again specifying points

    # in the top-left, top-right, bottom-right, and bottom-left

    # order

    dst = np.array([

        [0, 0],

        [maxWidth - 1, 0],

        [maxWidth - 1, maxHeight - 1],

        [0, maxHeight - 1]], dtype = "float32")



    # compute the perspective transform matrix and then apply it

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))



    # return the warped image

    return warped
# apply the four point transform to obtain a top-down

# view of the original image

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * 1/ratio)



# convert the warped image to grayscale, then threshold it

# to give it that 'black and white' paper effect

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

_,warped = cv2.threshold(warped, 200, 255, cv2.cv2.THRESH_TOZERO)



# show the original and scanned images

print("STEP 3: Apply perspective transform")

plt_imshow(cv2.resize(orig, (0,0), fy=ratio, fx=ratio))

plt_imshow(cv2.resize(warped, (0,0), fy=ratio, fx=ratio))