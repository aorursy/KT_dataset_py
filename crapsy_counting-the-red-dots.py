import numpy as np

import cv2

import matplotlib.pyplot as plt



img = cv2.cvtColor(cv2.imread("../input/TrainDotted/5.jpg"), cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap='gray')
# Mask everything but the red dots.

cmsk = cv2.inRange(img, np.array([160, 0, 0]), np.array([255, 50, 50])) # Get the red -ish stuff.

# Find the circles in the masked image.

circles = cv2.HoughCircles(cmsk,cv2.HOUGH_GRADIENT,1,50, param1=40,param2=1,minRadius=0,maxRadius=25)
#Draw rectangles around the dots. Print the total.

'''

if circles is not None:

    circles = np.uint16(np.around(circles))

    print('%d circles found.' % (len(circles[0]) ))

    for i in circles[0,:]:

        cv2.rectangle(img, (i[0] - 50, i[1] - 50), (i[0] + 50, i[1] + 50), (255, 0, 0), 3)



plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

'''
#Extract one sealion

i=0 #Sealion number

print(circles[0,i,:])

imgCrop= img[int(circles[0,i,1]-50):int(circles[0,i,1]+50),int(circles[0,i,0]-50):int(circles[0,i,0]+50)]

plt.imshow(imgCrop, cmap = 'gray', interpolation = 'bicubic')



#Convert to grayscal and normalize

imgCrop=cv2.cvtColor(imgCrop, cv2.COLOR_RGB2GRAY)

cv2.normalize(imgCrop, imgCrop, 0, 255, cv2.NORM_MINMAX)

plt.imshow(imgCrop, cmap = 'gray', interpolation = 'bicubic')
#Apply a canny filter to extract the edges

edges = cv2.Canny(imgCrop,50,50)

plt.imshow(edges,cmap = 'gray')
kernel = np.ones((5,5),np.uint8)

#dilation = cv2.dilate(edges,kernel,iterations = 1)

dilation = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

plt.imshow(dilation,cmap = 'gray')
#Fill function

def fillEdgeImage(img):

    edgesNeg = img.copy();

    # Mask used to flood filling.

    # Notice the size needs to be 2 pixels than the image.

    h, w = img.shape[:2]

    mask = np.zeros((h+2, w+2), np.uint8)

 

    # Floodfill from point (0, 0)

    cv2.floodFill(edgesNeg, mask, (0,0), 255);

    cv2.bitwise_not(edgesNeg, edgesNeg);

    filledImg = (edgesNeg | img);

    return filledImg
filledImg = fillEdgeImage(dilation)

plt.imshow(filledImg,cmap = 'gray')


# Setup SimpleBlobDetector parameters.

params = cv2.SimpleBlobDetector_Params()



# To extract white blobs

params.blobColor = 255;

# Change thresholds

params.minThreshold = 10

params.maxThreshold = 200

# Filter by Area.

params.filterByArea = True

params.minArea = 0

# Filter by Circularity

params.filterByCircularity = False

params.minCircularity = 0.1

# Filter by Convexity

params.filterByConvexity = False

params.minConvexity = 0.87

# Filter by Inertia

params.filterByInertia = False

params.minInertiaRatio = 0.01

# Create a detector with the parameters

detector = cv2.SimpleBlobDetector_create(params)



# Detect blobs.

keypoints = detector.detect(filledImg)

print(keypoints)

# Draw detected blobs as red circles.

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures

# the size of the circle corresponds to the size of blob



point1=keypoints[0].pt

print(point1)

print(keypoints[0].size)

'''

# Draw detected blobs as red circles.

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# Show keypoints

cv2.imshow("Keypoints", im_with_keypoints)

'''
#Find the blob that is closer to the center of the image

h, w = filledImg.shape

count=0;

euclDistPre=h**2+w**2;

for i in keypoints:

    hdiff=h/2-i.pt[0]

    wdiff=w/2-i.pt[1]

    euclDist = hdiff**2 + wdiff**2

    if euclDist<euclDistPre:

        euclDistPre = euclDist;

        blobNb=count;

    count+=1;   

#Define the area arround the sealion

print(keypoints[blobNb].size)

size=keypoints[blobNb].size*1.5

cv2.rectangle(filledImg, (round(h/2 - size/2),round(w/2 -  size/2)), (round(h/2 +  size/2), round(w/2 +  size/2)), (255, 0, 0), 3)

plt.imshow(filledImg,cmap = 'gray')
#crop the sealion

imgCropped=filledImg[round(h/2 - size/2):round(h/2 + size/2),round(w/2 -  size/2):round(w/2 +  size/2)]

plt.imshow(imgCropped,cmap = 'gray')