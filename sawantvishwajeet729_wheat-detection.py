## predict
import numpy as np

import time

import cv2

import os

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline
def display_img(img,cmap=None):

    fig = plt.figure(figsize = (12,12))

    plt.axis(False)

    ax = fig.add_subplot(111)

    ax.imshow(img,cmap)
labelsPath = "../input/model-files/yolo.names"

LABELS = open(labelsPath).read().strip().split("\n")
LABELS
from tensorflow.keras.layers import Input, Conv2D,ReLU, Flatten, Dense, MaxPool2D, concatenate, BatchNormalization, AveragePooling2D, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Activation
def inception(layer_in, num_filter):

    conv = Conv2D(num_filter,(3,3),padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

    conv2 = Conv2D(num_filter,(5,5),padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

    

    max_pool = MaxPool2D(pool_size=(2,2), strides=(1,1), padding='same')(layer_in)

    

    output = concatenate(inputs=[conv, conv2, max_pool], axis=-1)

    output = Activation('relu')(output)

    return output
def create_inception():

    inputs = Input(shape=(100,100,3))

    num_filters = 64

    

    t = BatchNormalization()(inputs)

    t = Conv2D(kernel_size=3, strides=1, filters=num_filters, padding="same")(t)

    t = Activation('relu')(t)

    

    t = inception(t,64)

    t= inception(t,32)

    

    t = AveragePooling2D(4)(t)

    t = Flatten()(t)

    

    outputs = Dense(7, activation='softmax')(t)

    

    model = Model(inputs, outputs)

    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    

    return model
model = create_inception()
model.load_weights("../input/model-files/wheat_detect.h5")
# derive the paths to the YOLO weights and model configuration

weightsPath = "../input/model-files/yolov3_custom_train_3000.weights"

configPath = "../input/modelfiles/yolov3_custom_train.cfg"
# Loading the neural network framework Darknet (YOLO was created based on this framework)

net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
# Create the function which predict the frame input

def predict(image):

    

    # initialize a list of colors to represent each possible class label

    np.random.seed(42)

    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    (H, W) = image.shape[:2]

    

    # determine only the "ouput" layers name which we need from YOLO

    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    

    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 

    # giving us our bounding boxes and associated probabilities

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)

    

    boxes = []

    confidences = []

    classIDs = []

    threshold = 0.2

    

    # loop over each of the layer outputs

    for output in layerOutputs:

        # loop over each of the detections

        for detection in output:

            # extract the class ID and confidence (i.e., probability) of

            # the current object detection

            scores = detection[5:]

            classID = np.argmax(scores)

            confidence = scores[classID]



            # filter out weak predictions by ensuring the detected

            # probability is greater than the minimum probability

            # confidence type=float, default=0.5

            if confidence > threshold:

                # scale the bounding box coordinates back relative to the

                # size of the image, keeping in mind that YOLO actually

                # returns the center (x, y)-coordinates of the bounding

                # box followed by the boxes' width and height

                box = detection[0:4] * np.array([W, H, W, H])

                (centerX, centerY, width, height) = box.astype("int")



                # use the center (x, y)-coordinates to derive the top and

                # and left corner of the bounding box

                x = int(centerX - (width / 2))

                y = int(centerY - (height / 2))



                # update our list of bounding box coordinates, confidences,

                # and class IDs

                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))

                classIDs.append(classID)



    # apply non-maxima suppression to suppress weak, overlapping bounding boxes

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)



    # ensure at least one detection exists

    if len(idxs) > 0:

        # loop over the indexes we are keeping

        for i in idxs.flatten():

            # extract the bounding box coordinates

            (x, y) = (boxes[i][0], boxes[i][1])

            (w, h) = (boxes[i][2], boxes[i][3])



            # draw a bounding box rectangle and label on the image

            color = (255,0,0)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            text = "{}".format(LABELS[0], confidences[i])

            cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)



    return image, boxes

# Execute prediction on a single image

img = cv2.imread("../input/global-wheat-detection/test/2fd875eaa.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

image, boxes = predict(img)

display_img(image)
# Create the function which predict the frame input

def predict(image):

    

    # initialize a list of colors to represent each possible class label

    np.random.seed(42)

    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    (H, W) = image.shape[:2]

    

    # determine only the "ouput" layers name which we need from YOLO

    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    

    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 

    # giving us our bounding boxes and associated probabilities

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)

    

    boxes = []

    confidences = []

    classIDs = []

    threshold = 0.2

    

    # loop over each of the layer outputs

    for output in layerOutputs:

        # loop over each of the detections

        for detection in output:

            # extract the class ID and confidence (i.e., probability) of

            # the current object detection

            scores = detection[5:]

            classID = np.argmax(scores)

            confidence = scores[classID]



            # filter out weak predictions by ensuring the detected

            # probability is greater than the minimum probability

            # confidence type=float, default=0.5

            if confidence > threshold:

                # scale the bounding box coordinates back relative to the

                # size of the image, keeping in mind that YOLO actually

                # returns the center (x, y)-coordinates of the bounding

                # box followed by the boxes' width and height

                box = detection[0:4] * np.array([W, H, W, H])

                (centerX, centerY, width, height) = box.astype("int")



                # use the center (x, y)-coordinates to derive the top and

                # and left corner of the bounding box

                x = int(centerX - (width / 2))

                y = int(centerY - (height / 2))



                # update our list of bounding box coordinates, confidences,

                # and class IDs

                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))

                classIDs.append(classID)



    # apply non-maxima suppression to suppress weak, overlapping bounding boxes

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)



    # ensure at least one detection exists

    if len(idxs) > 0:

        # loop over the indexes we are keeping

        for i in idxs.flatten():

            # extract the bounding box coordinates

            (x, y) = (boxes[i][0], boxes[i][1])

            (w, h) = (boxes[i][2], boxes[i][3])



            # draw a bounding box rectangle and label on the image

            color = (255,0,0)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            #text = "{}".format(LABELS[classIDs[i]], confidences[i])

            #cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)



            x1 = max(0,x)

            y1 = max(0,y)

            w1 = max(0,w)

            h1 = max(0,h)

            new_img = image[y1:y1+h1,x1:x1+w1]

            new_img = cv2.resize(new_img, (100,100))

            new_img = new_img.reshape(1,100,100,3)

            pred = model.predict(new_img)

            ids = np.argmax(pred, axis=1)

            if ids == 0:

                text = ('arvalis_1')

            elif ids == 1:

                text = ('arvalis_2')

            elif ids == 2:

                text = ('arvalis_3')

            elif ids == 3:

                text = ('ethz_1')

            elif ids == 4:

                text = ('inrae_1')

            elif ids == 5:

                text = ('rres_1') 

            else:

                text = ('usask_1')

            cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)

    return image

# Execute prediction on a single image

img = cv2.imread("../input/global-wheat-detection/test/796707dd7.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

image  = predict(img)

display_img(image)
# Create the function which predict the frame input

def predict_list(image):

    pred_string =[]

    

    # initialize a list of colors to represent each possible class label

    np.random.seed(42)

    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    (H, W) = image.shape[:2]

    

    # determine only the "ouput" layers name which we need from YOLO

    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    

    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 

    # giving us our bounding boxes and associated probabilities

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)

    

    boxes = []

    confidences = []

    classIDs = []

    threshold = 0.2

    

    # loop over each of the layer outputs

    for output in layerOutputs:

        # loop over each of the detections

        for detection in output:

            # extract the class ID and confidence (i.e., probability) of

            # the current object detection

            scores = detection[5:]

            classID = np.argmax(scores)

            confidence = scores[classID]



            # filter out weak predictions by ensuring the detected

            # probability is greater than the minimum probability

            # confidence type=float, default=0.5

            if confidence > threshold:

                # scale the bounding box coordinates back relative to the

                # size of the image, keeping in mind that YOLO actually

                # returns the center (x, y)-coordinates of the bounding

                # box followed by the boxes' width and height

                box = detection[0:4] * np.array([W, H, W, H])

                (centerX, centerY, width, height) = box.astype("int")



                # use the center (x, y)-coordinates to derive the top and

                # and left corner of the bounding box

                x = int(centerX - (width / 2))

                y = int(centerY - (height / 2))



                # update our list of bounding box coordinates, confidences,

                # and class IDs

                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))

                classIDs.append(classID)



    # apply non-maxima suppression to suppress weak, overlapping bounding boxes

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)



    # ensure at least one detection exists

    if len(idxs) > 0:

        # loop over the indexes we are keeping

        for i in idxs.flatten():

            # extract the bounding box coordinates

            (x, y) = (boxes[i][0], boxes[i][1])

            (w, h) = (boxes[i][2], boxes[i][3])



            # draw a bounding box rectangle and label on the image

            color = (255,0,0)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            #text = "{}".format(LABELS[classIDs[i]], confidences[i])

            #cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)



            x1 = max(0,x)

            y1 = max(0,y)

            w1 = max(0,w)

            h1 = max(0,h)

            new_img = image[y1:y1+h1,x1:x1+w1]

            new_img = cv2.resize(new_img, (100,100))

            new_img = new_img.reshape(1,100,100,3)

            pred = model.predict(new_img)

            ids = np.argmax(pred, axis=1)

            if ids == 0:

                text = ('arvalis_1')

            elif ids == 1:

                text = ('arvalis_2')

            elif ids == 2:

                text = ('arvalis_3')

            elif ids == 3:

                text = ('ethz_1')

            elif ids == 4:

                text = ('inrae_1')

            elif ids == 5:

                text = ('rres_1') 

            else:

                text = ('usask_1')

                

                

            str_id = ids[0]

            list_pred=[]

            list_pred.append(f"{str_id} {x1} {y1} {h1} {w1}")

            pred_string.append(list_pred)

             

    return pred_string

# Execute prediction on a single image

img = cv2.imread("../input/global-wheat-detection/train/00b70a919.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

list_1 = predict_list(img)
from os import listdir

from os.path import isfile, join

image_path = '../input/global-wheat-detection/test/'

onlyfiles = [f for f in listdir(image_path) if isfile(join(image_path, f))]
submission = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

submission = submission[0:0]
image_id = []

PredictionString =[]

for i in onlyfiles:

    path = (image_path+i)

    img = cv2.imread(path)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    list_1 = predict_list(img)

    res = [''.join(ele) for ele in list_1]

    res = ' '.join(res)

    image_id.append(i.strip('.jpg'))

    PredictionString.append(res)

    
submission['image_id'] = image_id

submission['PredictionString'] = PredictionString
submission.to_csv('submission.csv',index=False)
submission