address = '../input/face-detection-in-images/face_detection.json'
import json
import codecs
# get links and stuff from json

jsonData = []

with codecs.open(address, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

print(f"{len(jsonData)} image found!")

print("Sample row:")

jsonData[0]
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
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
import matplotlib.pyplot as plt
face1 = cv2.imread("./face-detection-images/face_image_64.jpg")
plt.figure(figsize=(20,25))
plt.imshow(face1)
plt.show()
plt.figure(figsize=(18,15))
plt.imshow(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))
face2 = cv2.imread("./face-detection-images/face_image_400.jpg")
plt.figure(figsize=(20,25))
plt.imshow(face2)
plt.show()