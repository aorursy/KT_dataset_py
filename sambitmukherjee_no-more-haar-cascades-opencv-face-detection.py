from fastai.vision import *

import numpy as np

import matplotlib.pyplot as plt

import cv2



%matplotlib inline
folder_path = Path('../input/5-celebrity-faces-dataset/train/madonna')
image_path = get_image_files(folder_path)[0]



image_path
image = cv2.imread(str(image_path))



image.shape
h, w = image.shape[:2]
plt.figure(figsize=(w/30, h/30))

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
arch = '../input/caffe-face-detector-opencv-pretrained-model/architecture.txt'

weights = '../input/caffe-face-detector-opencv-pretrained-model/weights.caffemodel'
neural_net = cv2.dnn.readNetFromCaffe(arch, weights)
blob = cv2.dnn.blobFromImage(

    image=cv2.resize(image, (299, 299)), # Resize the image to 299px by 299px.

    scalefactor=1.0, # Set the scaling factor.

    size=(299, 299), # Specify the spatial size of the image.

    mean=(103.93, 116.77, 123.68) # Normalize by subtracting the per-channel means of ImageNet images (which were used to train the pre-trained model).

)
neural_net.setInput(blob)

detections = neural_net.forward()
type(detections)
detections.shape
threshold = 0.5
for i in range(0, detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > threshold:

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        startX, startY, endX, endY = box.astype('int')

        text = '{:.2f}%'.format(confidence * 100)

        textY = startY - 10 if startY - 10 > 10 else startY + 10 # Ensure that the text won't go off-image.

        cv2.rectangle(

            img=image, 

            pt1=(startX, startY), # Vertex of the rectangle.

            pt2=(endX, endY), # Vertex of the rectangle opposite to `pt1`.

            color=(255, 0, 0),

            thickness=2

        )

        cv2.putText(

            img=image, 

            text=text, 

            org=(startX, textY), # Bottom-left corner of the text string.

            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 

            fontScale=0.5, 

            color=(255, 0, 0),

            thickness=2

        )
plt.figure(figsize=(w/30, h/30))

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))