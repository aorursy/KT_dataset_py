import numpy as np

import cv2

import matplotlib.pyplot as plt



# Filter ranges for each dot color.

redRange = [np.array([160, 0, 0]), np.array([255, 50, 50])]

magnetaRange = [np.array([160, 0, 160]), np.array([255, 50, 255])]

brownRange = [np.array([76, 39, 5]), np.array([94, 53, 22])]

blueRange = [np.array([0, 0, 160]), np.array([56, 56, 255])]

greenRange = [np.array([0, 160, 0]), np.array([56, 255, 56])]

colorRanges = [redRange, magnetaRange, brownRange, blueRange, greenRange ]



red = (255, 0, 0)

magneta = (255, 0, 255)

brown = (78, 42, 8)

blue = (0, 0, 255)

green = (0, 255, 0)

colors = [red, magneta, brown, blue, green]



img = cv2.cvtColor(cv2.imread("../input/TrainDotted/1.jpg"), cv2.COLOR_BGR2RGB)

imgPaint = img.copy()
counts = np.zeros(5)

for color in range(0,5):

    cmsk = cv2.inRange(img, colorRanges[color][0], colorRanges[color][1])

    circles = cv2.HoughCircles(cmsk,cv2.HOUGH_GRADIENT,1,50, param1=40,param2=1,minRadius=2,maxRadius=6)



    if circles is not None:

        circles = np.uint16(np.around(circles))

        counts[color] = len(circles[0,:])

        for i in circles[0,:]:

            cv2.rectangle(imgPaint, (i[0] - 50, i[1] - 50), (i[0] + 50, i[1] + 50), colors[color], 3)
print(counts)

plt.imshow(imgPaint, cmap = 'gray', interpolation = 'bicubic')