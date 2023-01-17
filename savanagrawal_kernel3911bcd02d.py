import cv2
import numpy as np
import time
net = cv2.dnn.readNetFromDarknet("../input/helmet-detection-yolov3/yolov3-helmet.cfg","../input/helmet-detection-yolov3/yolov3-helmet.weights")
classes = []
with open("../input/helmet-detection-yolov3/helmet.names","r") as f:
   classes = [line.strip() for line in f.readlines()]
print(classes)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(outputlayers)

