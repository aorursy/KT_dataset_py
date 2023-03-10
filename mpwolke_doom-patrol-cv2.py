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
from fastai.vision import *

import numpy as np

import matplotlib.pyplot as plt

import cv2



%matplotlib inline
folder_path = Path('../input/cusersmarilonedriveimagensdoomjpg')
image_path = get_image_files(folder_path)[0]



image_path
image = cv2.imread(str(image_path))



image.shape
h, w = image.shape[:2]
plt.figure(figsize=(w/30, h/30))

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
arch = '../input/caffe-face-detector-opencv-pretrained-model/architecture.txt'

weights = '../input/caffe-face-detector-opencv-pretrained-model/weights.caffemodel'
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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