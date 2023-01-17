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
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
names=open("/kaggle/input/yolo-coco-data/coco.names").read()
names
names=names.strip().split("\n")
print(names)
print(len(names))
weights_path = '/kaggle/input/yolo-coco-data/yolov3.weights'
configuration_path = '/kaggle/input/yolo-coco-data/yolov3.cfg'

pro_min = 0.5 # Setting minimum probability to eliminate weak predictions

threshold = 0.3 # Setting threshold for non maximum suppression
net = cv2.dnn.readNetFromDarknet(configuration_path,weights_path)

# Getting names of all layers
layers = net.getLayerNames()  # list of layers' names

# # Check point
print(layers)
for i in net.getUnconnectedOutLayers():
    print(layers[i[0]-1])
output_layers=[layers[i[0] - 1] for i in net.getUnconnectedOutLayers()] # We are searching for unconnected layers as output layers are not connected with any layer.

print(output_layers)
image=cv2.imread("/kaggle/input/imager1/image4.jpg")
image1=cv2.imread("/kaggle/input/imager/image2.jpeg")
print(image.shape)
# print(image1.shape)
plt.rcParams['figure.figsize'] = (8,8)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (400,400), swapRB=True, crop=False)
# blobfromimage returns a 4 dimensional bolb

# Check point
print(image.shape)  
print(blob.shape)  
# Resulted shape has number of images, number of channels, width and height
# So (1,3,400,400)
# Slicing blob and transposing to make channels come at the end
blob_to_show = blob[0, :, :, :].transpose(1, 2,0)
print(blob_to_show.shape) 

plt.rcParams['figure.figsize'] = (5, 5)
plt.imshow(blob_to_show)
plt.show()
print(blob.shape)
net.setInput(blob) # giving blob as input to our YOLO Network.
t1=time.time()
output = net.forward(output_layers)
t2 = time.time()

# Showing spent time for forward pass
print('YOLO took {:.5f} seconds'.format(t2-t1))
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416,416), swapRB=True, crop=False)
# blobfromimage returns a 4 dimensional bolb

# Check point
print(image.shape)  
print(blob.shape)  
# Resulted shape has number of images, number of channels, width and height
# Slicing blob and transposing to make channels come at the end
blob_to_show = blob[0, :, :, :].transpose(1, 2,0)
print(blob_to_show.shape) 

plt.rcParams['figure.figsize'] = (5, 5)
plt.imshow(blob_to_show)
plt.show()
print(blob.shape)
net.setInput(blob) # giving blob as input to our YOLO Network.
t1=time.time()
output = net.forward(output_layers)
t2 = time.time()

# Showing spent time for forward pass
print('YOLO took {:.2f} seconds'.format(t2-t1))
print(output)
print(output[0][0])
a=np.array([1,2,3,4,5,6,7])
print(a[2:])
print(a[0:4])
# for out in output:
#     for res in out:
#          print(res[5:])
colours = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8') # randint(low, high=None, size=None, dtype='l')

print(colours.shape)
print(len(colours))
print(colours[0])  
classes = []
confidences = []
boxes = []
Height = image.shape[0]
Width = image.shape[1]
print(Width,Height)
len(output)
for out in output:
    print(len(out))
# for out in output:
#     for res in out:
#         print(len(res))  
#     print('************************')
# for out in output:
#     for res in out:
#         print(len(res[5:]))  
#     print('************************')
# for out in output:
#      for res in out:
        
#         # print(res)
#         # Getting class for current object
#         scores = res[5:]
#         print(res[0])
for out in output:
    for res in out:
        
#         print(res)
        scores = res[5:]
#         print(scores)
        class_current = np.argmax(scores) # returning indices with max score and that would be our class as that will be 1 and rest will be 0

        # Getting the probability for current object by accessing the indices returned by argmax.
        confidence_current = scores[class_current]

        # Eliminating the weak predictions that is with minimum probability and this loop will only be encountered when an object will be there
        if confidence_current > 0.5:
            
            # Scaling bounding box coordinates to the initial image size
            # YOLO data format just keeps center of detected box and its width and height
            #that is why we are multiplying them elemwnt wise by width and height
            box = res[0:4] * np.array([Width, Height, Width, Height])  #In the first 4 indices only contains 
            #the output consisting of the coordinates.
            print(res[0:4])
            print(box)

            # From current box with YOLO format getting top left corner coordinates
            # that are x and y
            x, y, w, h = box.astype('int')
            x = int(x - (w / 2))
            y = int(y - (h / 2))
            

            # Adding results into the lists
            boxes.append([x, y, int(w), int(h)]) ## appending all the boxes.
            confidences.append(float(confidence_current)) ## appending all the confidences
            classes.append(class_current) ## appending all the classes         
results = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)

# Showing labels of the detected objects
for i in range(len(classes)):
    print(names[int(classes[i])])
results
results.flatten()
if len(results) > 0:

    for i in results.flatten():
        
        # Getting current bounding box coordinates
        x, y = boxes[i][0],boxes[i][1]
        width, height = boxes[i][2], boxes[i][3]
        
        colour_box_current = [int(j) for j in colours[classes[i]]]

        # Drawing bounding box on the original image
        cv2.rectangle(image, (x, y), (x + width, y + height),
                      colour_box_current, 2)

        # Preparing text with label and confidence 
        text_box_current = '{}: {:.4f}'.format(names[int(classes[i])], confidences[i])

        # Putting text with label and confidence
        cv2.putText(image, text_box_current, (x+2, y+20), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,0))
plt.rcParams['figure.figsize'] = (10,10)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()