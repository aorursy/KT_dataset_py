import numpy as np

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
img = cv2.imread('../input/russian_plate.jpg')

plt.imshow(img)
#I will make a bit larger so you can compare this with the final swapped plate image

fig, ax = plt.subplots(figsize=(20,20))

ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pretrained = '../input/haarcascade_russian_plate_number.xml'
#converting the image to grayscale is needed for detection by the classifier

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

#create a trained classifier object with the xml file

plate_cascade = cv2.CascadeClassifier(pretrained)



#detect the plate with the classifier

plate = plate_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 5, minSize = (40,40))

    

#opting to print bounding boxes to console; this is useful for the plates defined later on -- will hard code this manually

print(plate)
#use count to keep track of which are plates I want to swap

#the order of the plates detected is random each time you run this notebook; this is why I will manually hard code this later on

count = 0



#draw a rectangle around each detected plate with the list of returned bounding box coordinates

#don't want to overwrite the original image

detected_img = img



for (x, y, w, h) in plate:

    cv2.rectangle(detected_img, (x,y), (x + w, y + h), (0,0,255), 2) #red bounding box

    cv2.putText(detected_img, str(count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    count+=1

    

#let's see what plates were detected; remember to reverse the color order

plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
#plate of car on left has coordinates [ 291  435  133   45]

#plate of car on right has coordintes [ 875  414  137   46]



#here is the hard coding I mentioned previously

swap_this_plate_coords = [875,414,137,46]

with_this_plate_coords = [291,435,133,45]
#don't want to overwrite the original image; create a new image

img = cv2.imread('../input/russian_plate.jpg')

swap_img = img



#dims will be the width and height of swap_this_plate

dims = (137,46)



#slice out the part of the image that is with_this_plate; hard coded using with_this_plate_coords

#resize with_this_plate to be the same size as swap_this_plate (= dims); swap_this_plate has slightly larger dims

with_this_plate = img[with_this_plate_coords[1]:(with_this_plate_coords[1] + with_this_plate_coords[3]), with_this_plate_coords[0]:(with_this_plate_coords[0] + with_this_plate_coords[2]), :]

with_this_plate = cv2.resize(with_this_plate, dims, interpolation = cv2.INTER_AREA)



#these are swap_this_plate pixel areas to replace; hard coded using swap_this_plate_coords

for i,x in enumerate(range(swap_this_plate_coords[1],(swap_this_plate_coords[1] + swap_this_plate_coords[3]))):

    for j,y in enumerate(range(swap_this_plate_coords[0],(swap_this_plate_coords[0] + swap_this_plate_coords[2]))):

        swap_img[x,y,:] = with_this_plate[i,j,:]
#put some text to identify which plate was swapped

cv2.putText(swap_img, 'SWAPPED!', (swap_this_plate_coords[0],swap_this_plate_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



#let's see our image with the plate swapped

#I'll make the image a bit larger so you can see

fig, ax = plt.subplots(figsize=(20,20))

ax.imshow(cv2.cvtColor(swap_img, cv2.COLOR_BGR2RGB))