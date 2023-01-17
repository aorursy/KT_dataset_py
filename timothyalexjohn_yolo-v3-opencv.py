import numpy as np



import matplotlib.pyplot as plt

import cv2
classes = '../input/yolo-coco-data/coco.names'

config = '../input/yolo-coco-data/yolov3.cfg'

weights = '../input/yolo-coco-data/yolov3.weights'



LABELS = open(classes).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")    # Just to Colour the boxes
model = cv2.dnn.readNet(config, weights)     # 'readNet()' automatically directs the program to 'readNetFromDarknet(config, weights)'
img = cv2.imread('../input/data-for-yolo-v3-kernel/dog.jpg')         # Change Me!

(H, W) = img.shape[:2]

ln = model.getLayerNames()

output_ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]



img_blob = cv2.dnn.blobFromImage(img, 1/255.0, (608,608))            # Re-Scaling is IMPORTANT

model.setInput(img_blob)

outputs = model.forward(output_ln)  
for i in outputs:

    print(i.shape)        
THRESHOLD = 0.3

boxes = []

confidences = []

classIDs = []
for output in outputs[0]:

    scores = output[5:]

    classID = np.argmax(scores)

    confidence = scores[classID]



    if confidence > THRESHOLD:

        

        x = (output[0] - output[2]/2) * W        # location in image = position of anchor_box * image_pixel value

        y = (output[1] - output[3]/2) * H



        boxes.append([int(x), int(y), int(output[2]* W), int(output[3]* H)])

        confidences.append(float(confidence))

        classIDs.append(classID)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, THRESHOLD, THRESHOLD)



for idx in idxs:

    idx = idx[0]

    color = [int(c) for c in COLORS[classIDs[idx]]]

    cv2.rectangle(img, (boxes[idx][0],boxes[idx][1]), (boxes[idx][0]+boxes[idx][2],boxes[idx][1]+boxes[idx][3]), color)

    cv2.putText(img, "{}  {:.2f}".format(LABELS[classIDs[idx]], confidences[idx]), (boxes[idx][0],boxes[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)



cv2.imwrite("example.png", img)

plt.imshow(img)