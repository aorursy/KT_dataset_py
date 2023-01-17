from IPython.display import Image

Image('../input/self-driving-cars/images/1478020200190518278.jpg')
import numpy as np

import pandas as pd

import cv2



from sklearn.utils import shuffle

from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
df = pd.read_csv('../input/self-driving-cars/labels_train.csv')

df = shuffle(df)

df.head()
classes = df.class_id.unique()

print(classes)
labels = { 1:'car', 2:'truck', 3:'pedestrian', 4:'bicyclist', 5:'light'}
# Get path images and boxes (x,y) for each class_id

boxes = {}

images = {}



base_path = '../input/self-driving-cars/images/'



for class_id in classes:

    first_row = df[df['class_id'] == class_id].iloc[0]

    

    images[class_id] = cv2.imread(base_path + first_row['frame'])

    boxes[class_id] = [first_row['xmin'],first_row['xmax'],first_row['ymin'],first_row['ymax']]
boxes
for i in classes:



    xmin, xmax, ymin, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]



    plt.figure(figsize=(10, 10))

    plt.title("Label " + labels[i])

    plt.imshow(images[i])

    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))

    

    plt.show()