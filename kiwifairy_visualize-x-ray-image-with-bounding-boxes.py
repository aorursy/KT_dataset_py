import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import os
path = '../input/'

class_path = path + 'stage_1_detailed_class_info.csv'

label_path = path + 'stage_1_train_labels.csv'

train_img_path = path + 'stage_1_train_images/'

test_img_path = path + 'stage_1_test_images'
label = pd.read_csv(label_path)
classes = pd.read_csv(class_path)
print(sum(label['Target']==1)/label.shape[0])  # data imbalance
sum(label['Target']==1)
class_set = set(classes['class'])  # Lung Opacity = 1, otherwise = 0
sum(classes['class'] == 'Lung Opacity') == sum(label['Target']==1)
train_img = []
for root, dirs, files in os.walk(train_img_path):
    for name in files:
        if name[-3:] == 'dcm':
            train_img.append(name)
        
def plot_img_bbox(img, boxes_df):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(10,10)
    a.imshow(img, cmap = 'gray')
    for index, row in boxes_df.iterrows():
        x, y, width, height  = row['x'], row['y'], row['width'], row['height']
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()
def look_img(img_class=None, pID=None):
    if pID == None:
        if img_class == None:
            pID = classes['patientId'].sample(1).iloc[0]
            img_class = classes[classes['patientId']==pID]['class']
        else:
            pID = classes[classes['class'] == img_class]['patientId'].sample(1).iloc[0]
    dicom = pydicom.dcmread(train_img_path+pID+'.dcm')
    # get the image pixels
    img = dicom.pixel_array
    # get the bboxes, each box is one row
    boxes_df = label[label['patientId'] == pID]
    plot_img_bbox(img, boxes_df)
    # get age and gender
    age = int(dicom.PatientAge)
    gender = dicom.PatientSex
    print(img_class, '\n', age, gender, pID)
for cls in class_set:
    for i in range(5):
        look_img(cls)
# check how many bbox max in a img
from collections import Counter
c = Counter(label['patientId'])
sort = sorted(c.items(), key=lambda item: item[1], reverse=True)
sort
look_img(pID = sort[0][0])
look_img(pID = sort[15][0])