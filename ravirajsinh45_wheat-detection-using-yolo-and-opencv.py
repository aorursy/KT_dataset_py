import numpy as np

import pandas as pd

import argparse

import time

import cv2

import os

import matplotlib.pyplot as plt
sub = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')

sub
sub.iloc[0,1]
#load the COCO class labels our YOLO model was trained on

labelsPath = '/kaggle/input/global-wheat-detection-yolo-weights/obj.names'

LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),

    dtype="uint8")
weightsPath = '/kaggle/input/global-wheat-detection-yolo-weights/wheat_detection_tiny_last.weights'

configPath = '/kaggle/input/global-wheat-detection-yolo-weights/wheat_detection_tiny.cfg'
print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
test = '/kaggle/input/global-wheat-detection/test/'

test_img_list = os.listdir(test)
plt.figure(figsize=(24,15))

for i,image in enumerate(test_img_list):

    img = cv2.imread(test + image)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    plt.subplot(2,5,i+1)

    plt.imshow(img)

    plt.title(image[:-5])

    plt.axis('off')

    

plt.show()
sub.loc[0,'image_id']
plt.figure(figsize=(20,15))

for image_id in range(len(sub)):

    img = sub.loc[image_id,'image_id']

    image = cv2.imread(test+img+'.jpg')

    (H, W) = image.shape[:2]



    thresh = 0.2

    confi = 0.1

    pred_str = ''

    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #construct a blob from the input image and then perform a forward

    #pass of the YOLO object detector, giving us our bounding boxes and

    #associated probabilities

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512,512),swapRB=True, crop=False)

    net.setInput(blob)

    start = time.time()

    layerOutputs = net.forward(ln)

    end = time.time()



    #show timing information on YOLO

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))



    #initialize our lists of detected bounding boxes, confidences, and

    #class IDs, respectively

    boxes = []

    confidences = []

    classIDs = []



    #loop over each of the layer outputs

    for output in layerOutputs:

        #loop over each of the detections

        for detection in output:

            #extract the class ID and confidence (i.e., probability) of

            #the current object detection

            scores = detection[5:]

            classID = np.argmax(scores)

            confidence = scores[classID]



            #filter out weak predictions by ensuring the detected

            #probability is greater than the minimum probability

            if confidence > confi:

                #scale the bounding box coordinates back relative to the

                #size of the image, keeping in mind that YOLO actually

                #returns the center (x, y)-coordinates of the bounding

                #box followed by the boxes' width and height

                box = detection[0:4] * np.array([W, H, W, H])

                (centerX, centerY, width, height) = box.astype("int")



                #use the center (x, y)-coordinates to derive the top and left corner of the bounding box

                x = int(centerX - (width / 2))

                y = int(centerY - (height / 2))



                #update our list of bounding box coordinates, confidences,#and class IDs

                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))

                classIDs.append(classID)



    #apply non-maxima suppression to suppress weak, overlapping bounding

    #boxes

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi ,thresh)

    #ensure at least one detection exists

    if len(idxs) > 0:

        #loop over the indexes we are keeping

        for i in idxs.flatten():

            #extract the bounding box coordinates

            (x, y) = (boxes[i][0], boxes[i][1])

            (w, h) = (boxes[i][2], boxes[i][3])

            pred = '{} {} {} {} {} '.format(np.round(confidences[i],1),x,y,w,h)

            pred_str = pred_str + pred

            

            #draw a bounding box rectangle and label on the image

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)

    

    

    sub.loc[image_id,'PredictionString'] = pred_str[:-1]

    





    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    plt.subplot(2,5,image_id+1)

    plt.imshow(image)

    plt.title(img)

    plt.axis('off')

    

     

    

    

plt.show()  

sub
sub.to_csv('submission.csv',index=False)