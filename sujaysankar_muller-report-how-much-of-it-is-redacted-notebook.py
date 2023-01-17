# import the necessary packages

%matplotlib inline



from matplotlib import pyplot as plt



from imutils import contours

from skimage import measure

import numpy as np

import imutils

import cv2



# load the image, convert it to grayscale, and blur it

image = cv2.imread("../input/report-070.ppm.jpg")

print("Original Image")

plt.imshow(image)

plt.show()



print("Grayscale Image")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray)

plt.show()



blurred = cv2.GaussianBlur(gray, (11, 11), 0)

print("Blurred Image")

plt.imshow(blurred)

plt.show()



# threshold the image to reveal redacted regions in the image

thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

thresh = cv2.erode(thresh, None, iterations=2)

thresh = cv2.dilate(thresh, None, iterations=4)



print("Redactions highlighted")

plt.imshow(thresh)

plt.show()



labels = measure.label(thresh, neighbors=8, background=0)

mask = np.zeros(thresh.shape, dtype="uint8")



totalRedactedArea = 0



# loop over the unique components

for label in np.unique(labels):

    # if this is the background label, ignore it

    if label != 0:

        continue



    # otherwise, construct the label mask and count the

    # number of pixels

    labelMask = np.zeros(thresh.shape, dtype="uint8")

    labelMask[labels == label] = 255

    numPixels = cv2.countNonZero(labelMask)

    # print(numPixels)

    # if the number of pixels in the component is sufficiently

    # large, then add it to our mask of "large blobs"

    if numPixels > 300:

        mask = cv2.add(mask, labelMask)



# find the contours in the mask, then sort them from left to right

cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,

                        cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

cnts = contours.sort_contours(cnts)[0]



# loop over the contours

for (i, c) in enumerate(cnts):

    # draw the bright spot on the image

    (x, y, w, h) = cv2.boundingRect(c)

    print("Redacted Zone:", i, "Area: ", w*h)

    totalRedactedArea = totalRedactedArea + w*h

    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 5)

    cv2.putText(image, "#{}".format(w*h), (x, y - 15),

                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)







(height, width, channels) = image.shape

cX = 210

cY = 210



cv2.rectangle(image, (cX,cY), (width-cX,height-cY), (0,255,0), 5)

# show the output image

print("Final Result")

plt.imshow(image)

plt.show()



totalArea = (width-cX) * (height-cY)

pctRedacted = (totalRedactedArea / totalArea)*100



print("Summary:")

print("Total Area of Document:", totalArea)

print("Total redacted Area:", totalRedactedArea)

print("%Redaction:",pctRedacted)