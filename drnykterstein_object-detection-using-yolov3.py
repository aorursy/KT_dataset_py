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

import matplotlib.pyplot as plt


import cv2

import numpy as np



net = cv2.dnn.readNet('../input/yolo-coco-data/yolov3.weights', '../input/yolo-coco-data/yolov3.cfg')

cap = cv2.VideoCapture('../input/test-video/test.mp4')

classes = []

with open("../input/yolo-coco-data/coco.names", "r") as f:

    classes = f.read().splitlines()





font = cv2.FONT_HERSHEY_PLAIN

colors = np.random.uniform(0, 255, size=(100, 3))
img = cv2.imread('../input/input-2/Half-Page-Size-Laurel-Restricted.jpg')

len(classes)# number of label

#getting the op layers name and then using forward to extract the op in the layers
#modifing the input and the RGB channels

height,width,_  = img.shape

print(height,width)
#BGR mode for image 

plt.imshow(img)
#after converting the channels from BGR to BGR (real channels for image)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#getting blob from input image

blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB = True,crop = False)

print(blob.shape)#1 smaple  and 3 channels and width ans height 
import time
start = time.time()

net.setInput(blob)

#getting the op layer and then extract the stuff from it 

op_layer_name  = net.getUnconnectedOutLayersNames()

layer_op = net.forward(op_layer_name)

end = time.time()

print('YOLO v3 took {:.5f} seconds'.format(end - start))

# Preparing lists for detected bounding boxes, obtained confidences and class's number

bounding_boxes = []

confidences = []

class_numbers = []
import numpy as np
h = height

w = width
for result in layer_op:

    # Going through all detections from current output layer

    for detection in result:

        # Getting class for current object

        scores = detection[5:]

        class_current = np.argmax(scores)



        # Getting confidence (probability) for current object

        confidence_current = scores[class_current]



        # Eliminating weak predictions by minimum probability

        if confidence_current > 0.5:

            # Scaling bounding box coordinates to the initial image size

            # YOLO data format keeps center of detected box and its width and height

            # That is why we can just elementwise multiply them to the width and height of the image

            box_current = detection[0:4] * np.array([w, h, w, h])



            # From current box with YOLO format getting top left corner coordinates

            # that are x_min and y_min

            x_center, y_center, box_width, box_height = box_current.astype('int')

            x_min = int(x_center - (box_width / 2))

            y_min = int(y_center - (box_height / 2))



            # Adding results into prepared lists

            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])

            confidences.append(float(confidence_current))

            class_numbers.append(class_current)
len(bounding_boxes)#52 object has been detecting 
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)
print(results.flatten())
for i in results.flatten():

    x,y,w,h = bounding_boxes[i]

    label = str(classes[class_numbers[i]])

    confidence = str(round(confidences[i],2))

    color = colors[i] #random color

    #drawing rectangle and putting the text

    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

    cv2.putText(img,label+" "+confidence,(x,y+20),font,2,(255,255,255),2)

    
import matplotlib.pyplot as plt

import cv2
%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()
