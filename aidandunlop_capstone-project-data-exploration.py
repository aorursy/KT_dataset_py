# Import dependencies

import numpy as np

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import cv2

import random

import os

import numpy as np

import pandas as pd

import matplotlib.patches as patches

anno_path_box = "../input/sample-dayClip6/sample-dayClip6/frameAnnotationsBOX.csv"

anno_path_bulb = "../input/sample-dayClip6/sample-dayClip6/frameAnnotationsBULB.csv"

frames_path = "../input/sample-dayClip6/sample-dayClip6/frames/"

print("Choosing random image...")

frame_id = random.choice(os.listdir(frames_path))

# frame_id = 'dayClip6--00099.jpg'  

frame_path = os.path.join(os.path.join(frames_path,frame_id))

print("Using image:\t'{}'\nLocation:\t'{}'".format(frame_id, frame_path))

# To draw a box around the traffic light bulb, use 'anno_path_bulb' below, instead of 'anno_path_box'

annotations = []

with open(anno_path_box) as fp:  

    line = fp.readline()

    line = fp.readline() # Skip header line with descriptions

    #cnt = 1

    while line:

        anno_file_path = (line.strip()).split(";")

        anno_file_id = anno_file_path[0].split("/")[1]

        if anno_file_id == frame_id:

            annotations.append(anno_file_path)

            #print("\t{}".format(anno_file_id))

        line = fp.readline()
# Plot annotations on image

color_space = [(0,255,0),(255,0,0),(255,0,0)]

img = cv2.imread(frame_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.rcParams['figure.figsize'] = [32, 16]

plt.imshow(img)

plt.show()

print("Found {} annotations:".format(len(annotations)))



for anno in annotations:

    anno_class = anno[1]

    anno_left = int(anno[2])

    anno_top = int(anno[3])

    anno_right = int(anno[4])

    anno_bot = int(anno[5])

    print("\tClass: '{}' at [{},{},{},{}]".format(anno_class, anno_left, anno_top, anno_right, anno_bot))

    if anno_class == "go" or anno_class == "goLeft" or anno_class == "goForward":

        color_class = color_space[0]

    elif anno_class == "warning" or anno_class == "warningLeft":

        color_class = (255,255,0)

    elif anno_class == "stop" or anno_class == "stopLeft":

        color_class = color_space[1]

    cv2.rectangle(img, (anno_left, anno_top), (anno_right, anno_bot), color_class, 2)



    

plt.imshow(img)

plt.show()

anno_data_frame = pd.read_csv(anno_path_box, ';') 

anno_data_frame.head()
anno_data_frame.describe()
def getImageSize(frame_path):

    img = cv2.imread(frame_path)

    height, width, channels = img.shape

    return [height, width]
imageSizes = [getImageSize(os.path.join(frames_path,frame_id)) for frame_path in os.listdir(frames_path)]

result = set(map(tuple, imageSizes))

print("Number of unique frame sizes: {}".format(len(result)))
plt.figure(figsize=(32,16))

currentAxis = plt.gca()



for index, row in anno_data_frame.iterrows():

    width = row['Lower right corner X'] - row['Upper left corner X']

    height = row['Lower right corner Y'] - row['Upper left corner Y']

    topLeft = (row['Upper left corner X'], row['Upper left corner Y'])

    currentAxis.add_patch(patches.Rectangle(topLeft, width, height, alpha=1, fill=False, edgecolor='green'))



example_image = cv2.imread(frame_path)

example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)



plt.imshow(example_image)

plt.show()

anno_data_frame['Annotation tag'].value_counts()
anno_path_box2 = "../input/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv"

frames_path2 = "../input/daySequence1/daySequence1/frames/"

anno_data_frame2 = pd.read_csv(anno_path_box2, ';')

anno_data_frame2['Annotation tag'].value_counts()