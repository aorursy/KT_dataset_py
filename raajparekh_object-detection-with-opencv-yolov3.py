import numpy as np 
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
net = cv2.dnn.readNet("/kaggle/input/yolov3-weight/yolov3.weights", "/kaggle/input/yolov3-weight/yolov3.cfg")
layer_names = net.getLayerNames()
print("layers names:")
print(layer_names)
output_layers = net.getUnconnectedOutLayersNames()
print("output layers:")
print(output_layers)
classes = []
with open("/kaggle/input/coconames/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
colors = np.random.uniform(0, 255, size=(len(classes), 3)) #This will be used later to assign colors for the bounding box for the detected objects
def get_objects_predictions(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor = 1/255, size = (416, 416), mean= (0, 0, 0), swapRB = True, crop=False)
    net.setInput(blob)
    predictions = net.forward(output_layers)
    return predictions,height, width
def get_box_dimentions(predictions,height, width, confThreshold = 0.5):
    class_ids = []
    confidences = []
    boxes = []
    for out in predictions:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)#Identifing the class type of the detected object by checking maximum confidence
            confidence = scores[class_id]
            if confidence > confThreshold:
                # Object detected
                center_x = int(detection[0] * width) #converting center_x with respect to original image size
                center_y = int(detection[1] * height)#converting center_y with respect to original image size
                w = int(detection[2] * width)#converting width with respect to original image size
                h = int(detection[3] * height)#converting height with respect to original image size
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes,confidences,class_ids
def non_max_suppression(boxes,confidences,confThreshold = 0.5, nmsThreshold = 0.4):
    return cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
def draw_bouding_boxes(img,boxes,confidences,class_ids,nms_indexes,colors):
    for i in range(len(boxes)):
        if i in nms_indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) + ' :' + str(int(confidences[i]*100)) + '%'
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label, (x, y - 15),cv2.FONT_HERSHEY_PLAIN ,2, color, 3)
    return img
def detect_objects(img_path):
    predictions,height, width = get_objects_predictions(img_path)
    boxes,confidences,class_ids = get_box_dimentions(predictions,height, width)
    nms_indexes = non_max_suppression(boxes,confidences)
    img = draw_bouding_boxes(img_path,boxes,confidences,class_ids,nms_indexes,colors)
    return img
files = ['/kaggle/input/open-images-2019-object-detection/test/' + i for i in os.listdir('/kaggle/input/open-images-2019-object-detection/test')]
plt.figure(figsize=(25,30))

for i in range(1,13):
    index = np.random.randint(len(files))
    plt.subplot(6, 2, i)
    plt.imshow(detect_objects(cv2.imread(files[index])), cmap='cool')
plt.show()