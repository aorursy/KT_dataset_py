# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from ast import literal_eval

import cv2

import matplotlib.pyplot as plt





train_df = pd.read_csv("../input/global-wheat-detection/train.csv")
train_df.head()
def convert(size, box):

    dw = 1. / size[0]

    dh = 1. / size[1]

    x = (box[0] + box[1]) / 2.0

    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]

    h = box[3] - box[2]

    x = x * dw

    w = w * dw

    y = y * dh

    h = h * dh

    return [x, y, w, h]



def convert_to_yolo_label(coco_format_box, w = 1024, h = 1024):

    bbox = literal_eval(coco_format_box)

    xmin = bbox[0]

    xmax = bbox[0] + bbox[2]

    ymin = bbox[1]

    ymax = bbox[1] + bbox[3]

    b = (float(xmin), float(xmax), float(ymin), float(ymax))

    yolo_box = convert((w, h), b)

    if np.max(yolo_box) > 1 or np.min(yolo_box) < 0: # Take this opportunity to check that conversion works

        print("BOX HAS AN ISSUE")

    return yolo_box



%%time

train_df['yolo_box'] = train_df.bbox.apply(convert_to_yolo_label)
train_df.head()
def convertYoloToPascal(size, coord):

    x2 = int(((2*size[0]*float(coord[0]))+(size[0]*float(coord[2])))/2)

    x1 = int(((2*size[0]*float(coord[0]))-(size[0]*float(coord[2])))/2)



    y2 = int(((2*size[1]*float(coord[1]))+(size[1]*float(coord[3])))/2)

    y1 = int(((2*size[1]*float(coord[1]))-(size[1]*float(coord[3])))/2)

    return (x1,y1,x2,y2)



convertYoloToPascal((1024, 1024), [0.841796875, 0.234375, 0.0546875, 0.03515625])
## Check that we can visualize boxes as Coco format:

fr = cv2.imread("../input/global-wheat-detection/train/b6ab77fd7.jpg")

frame_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

frame_rgb = cv2.rectangle(frame_rgb, (834, 222), (834+56, 222+36), (255, 0, 0), 5)

plt.imshow(frame_rgb)
print("We have {} unique images with boxes.".format(len(train_df.image_id.unique())))

unique_img_ids = train_df.image_id.unique()
if not os.path.exists("wheat_yolo_train_data"):

    os.makedirs("wheat_yolo_train_data")
folder_location = "wheat_yolo_train_data"

#change  unique_img_ids[:2] to unique_img_ids to iterate through all images

for img_id in unique_img_ids[:2]: # loop through all unique image ids. Remove the slice to do all images

    print(img_id)

    filt_df = train_df.query("image_id == @img_id") # filter the df to a specific id

    #print(filt_df.shape[0])

    all_boxes = filt_df.yolo_box.values

    file_name = "{}/{}.txt".format(folder_location,img_id) # specify the name of the folder and get a file name



    s = "0 %s %s %s %s \n" # the first number is the identifier of the class. If you are doing multi-class, make sure to change that

    with open(file_name, 'a') as file: # append lines to file

        for i in all_boxes:

            new_line = (s % tuple(i))

            file.write(new_line)  

!ls wheat_yolo_train_data/
!head wheat_yolo_train_data/b53afdf5c.txt
import glob

all_imgs = glob.glob("../input/global-wheat-detection/train/*.jpg")

all_imgs = [i.split("/")[-1].replace(".jpg", "") for i in all_imgs]

positive_imgs = train_df.image_id.unique()
negative_images = set(all_imgs) - set(positive_imgs)

print(len(all_imgs), len(positive_imgs))
len(negative_images)
for i in list(negative_images)[:2]:

    file_name = "wheat_yolo_train_data/{}.txt".format(i)

    print(file_name)

    with open(file_name, 'w') as fp: 

        pass
!cat wheat_yolo_train_data/91d4e3750.txt