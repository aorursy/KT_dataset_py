# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.pyplot as plt

from IPython.display import clear_output, Image, display

%matplotlib inline
image =cv2.cvtColor(cv2.imread("../input/road-image-with-lanes/test_image.jpg"), cv2.COLOR_BGR2RGB) 



plt.imshow(image)

plt.show()
import numpy as np



lane_image = np.copy(image)

lane_image.shape
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)



plt.imshow(gray,cmap='gray')

plt.show()
blur = cv2.GaussianBlur(gray, (5,5), 0)



plt.imshow(blur)

plt.show()
canny = cv2.Canny(blur, 50,150)



plt.imshow(canny)

plt.show()
# put all the previous 3 steps in one function 

# gray scale conversion ==> applying gaussian blur ==> applying canny function



def Canny(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    canny = cv2.Canny(blur, 50, 150)

    

    return canny
image = cv2.imread("../input/road-image-with-lanes/test_image.jpg")

lane_image = np.copy(image)

canny = Canny(lane_image)



plt.imshow(canny)

plt.show()
# make a function to find region of interest. 

# then, apply bitwise AND operation in order to select just the region of interest



def region_of_interest(image):

    height = 700

    polygons = np.array([[(250,height), (1000,height), (600,250)]])

    mask = np.zeros_like(image)

    cv2.fillPoly(mask, polygons, 255)

    masked_region = cv2.bitwise_and(image, mask)

    

    return masked_region
# function which draws lines as detected lanes in the black image



def display_lines(image, lines):

    line_image = np.zeros_like(image)

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line.reshape(4)

            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)

    return line_image
# reading the image and then calling all the functions one by one. 



image = cv2.imread("../input/road-image-with-lanes/test_image.jpg")



lane_image = np.copy(image)

canny_image = Canny(lane_image)

cropped_image = region_of_interest(canny_image)



lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), 40, 5)

line_image = display_lines(lane_image, lines)



plt.imshow(line_image)

plt.show()
# final step: put it all together

# now combine the above immage containing just the lines in a black image with our original image



combo_image = cv2.addWeighted(lane_image, 1, line_image, 1, 1)



plt.imshow(combo_image)

plt.show()
def make_coordinates(image, line_parameters):

    slope, intercept = line_parameters

    y1 = image.shape[0]

    y2 = int(y1*(3/5))

    x1 = int((y1-intercept)/slope)

    x2 = int((y2-intercept)/slope)

    return np.array([x1,y1,x2,y2])
def average_slope_intercept(image, lines):

    left_fit = []

    right_fit = []

    

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        parameters = np.polyfit((x1,x2), (y1,y2), 1)

        

        slope = parameters[0]

        intercept = parameters[1]

        

        if slope < 0:

            left_fit.append((slope, intercept))

        elif slope >= 0:

            right_fit.append((slope, intercept))

            

    left_fit_average = np.average(left_fit, axis=0)

    right_fit_average = np.average(right_fit, axis=0)

        

    left_line = make_coordinates(image, left_fit_average)

    right_line = make_coordinates(image, right_fit_average)

        

    return np.array([left_line, right_line])
# reading the image and then calling all the functions one by one. 



image = cv2.imread("../input/road-image-with-lanes/test_image.jpg")



lane_image = np.copy(image)

canny_image = Canny(lane_image)

cropped_image = region_of_interest(canny_image)



lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), 40, 5)

averaged_lines = average_slope_intercept(lane_image, lines)

line_image = display_lines(lane_image, averaged_lines)



# final step: put it all together

# now combine the above immage containing just the lines in a black image with our original image



combo_image = cv2.addWeighted(lane_image, 1, line_image, 1, 1)



plt.imshow(combo_image)

plt.show()
cap = cv2.VideoCapture("../input/video-file-for-lane-detection-project/test_video.mp4")



#while (cap.isOpened()):

#    _, frame = cap.read()

#    canny_image = Canny(frame)

#    cropped_image = region_of_interest(canny_image)

#    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), 40, 5)

#    averaged_lines = average_slope_intercept(frame, lines)

#    line_image = display_lines(frame, averaged_lines)

#    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)

    

#    cv2.imshow("result", combo_image)

#    if cv2.waitKey(1) == ord("q"):

#        break

#cap.release()

#cv2.destroyAllWindows()