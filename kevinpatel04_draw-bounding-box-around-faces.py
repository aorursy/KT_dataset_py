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
# import libraries

import json

import codecs

import requests

import numpy as np

import pandas as pd 

from PIL import Image

from tqdm import tqdm

from io import BytesIO
# get links and stuff from json

jsonData = []

JSONPATH = "/kaggle/input/face-detection-in-images/face_detection.json"

with codecs.open(JSONPATH, 'rU', 'utf-8') as js:

    for line in js:

        jsonData.append(json.loads(line))



print(f"{len(jsonData)} image found!")



print("Sample row:")

jsonData[0]
# load images from url and save into images

images = []

for data in tqdm(jsonData):

    response = requests.get(data['content'])

    img = np.asarray(Image.open(BytesIO(response.content)))

    images.append([img, data["annotation"]])
!mkdir face-detection-images
import cv2

import time

count = 1

totalfaces = 0

start = time.time()

for image in images:

    img = image[0]

    metadata = image[1]

    for data in metadata:

        height = data['imageHeight']

        width = data['imageWidth']

        points = data['points']

        if 'Face' in data['label']:

            x1 = round(width*points[0]['x'])

            y1 = round(height*points[0]['y'])

            x2 = round(width*points[1]['x'])

            y2 = round(height*points[1]['y'])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            totalfaces += 1

    cv2.imwrite('./face-detection-images/face_image_{}.jpg'.format(count),img)

    count += 1

    

end = time.time()

print("Total test images with faces : {}".format(len(images)))

print("Sucessfully tested {} images".format(count-1))

print("Execution time in seconds {}".format(end-start))

print("Total Faces Detected {}".format(totalfaces))