# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from shutil import copyfile

#Copy files from data

copyfile(src = "../input/mtcnnmodel/face_detector.py", dst = "../working/face_detector.py")

copyfile(src = "../input/mtcnnmodel/faceConfig_MTCNN_config.py", dst = "../working/faceConfig_MTCNN_config.py")

copyfile(src = "../input/mtcnnmodel/faceDetection_nms.py", dst = "../working/faceDetection_nms.py")

copyfile(src = "../input/mtcnnmodel/facePrepareData_loader.py", dst = "../working/facePrepareData_loader.py")

copyfile(src = "../input/mtcnnmodel/faceDetection_detector.py", dst = "../working/faceDetection_detector.py")

copyfile(src = "../input/mtcnnmodel/facePrepareData_minibatch.py", dst = "../working/facePrepareData_minibatch.py")

copyfile(src = "../input/mtcnnmodel/faceDetection_fcn_detector.py", dst = "../working/faceDetection_fcn_detector.py")

copyfile(src = "../input/mtcnnmodel/faceConfig_mtcnn_model.py", dst = "../working/faceConfig_mtcnn_model.py")

copyfile(src = "../input/mtcnnmodel/faceDetection_mtcnnDetector.py", dst = "../working/faceDetection_mtcnnDetector.py")

import cv2

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt

from face_detector import FaceDetector

import os



#Model loading ...

faceDetector = FaceDetector('../input/pnet-landmark/PNet', '../input/rnet-landmark/RNet', '../input/onet-landmark/ONet')

#Loading image from input data

image = cv2.imread('../input/testcase/abc2.jpg')



#Implement face detecting on your image

results = faceDetector.predict([image])



#Change color channel

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



#Draw bounding box

for bbox in results[0]:

    height, width, _ = image.shape

    box = [int(bbox.xmin*width), int(bbox.ymin*height), int(bbox.xmax*width), int(bbox.ymax*height)]

    print(box)

    cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,255,0),3)



#Show result

plt.imshow(image)

plt.title('my picture')

plt.show()