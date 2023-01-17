# Importing libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import cv2

import warnings

import os

from tqdm import tqdm

warnings.filterwarnings('ignore')
train_csv_file = '../input/global-wheat-detection/train.csv'

train_images = '../input/global-wheat-detection/train/'
#Load dataframe

train_df = pd.read_csv(train_csv_file)
train_df.head()
# check for any null values

train_df.isnull().any().any()


# check for any height and width is other than 1024

print((train_df['height'] != 1024).any())

print((train_df['width'] != 1024).any())
len(train_df['image_id'].unique())
print('Number of images without label: {}'.format(len(os.listdir(train_images)) - len(train_df['image_id'].unique())))
# create new dataframe with only columns: image_id and bbox

img_bbox = train_df[['image_id','bbox']]
img_bbox.loc[0,'bbox']
# converting string into list

img_bbox['bbox'] = img_bbox['bbox'].str.strip('][').str.split(',')

img_bbox.head()
# Now it's in list datatype

img_bbox.loc[0,'bbox']
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
tfrecord_format_csv = pd.DataFrame(columns=column_name)



for i in tqdm(range(len(img_bbox))):

    img_bbox['bbox'][i] = pd.to_numeric(img_bbox['bbox'][i],downcast='integer')

    tfrecord_format_csv.loc[i,'filename'] = img_bbox.loc[i,'image_id']+ '.jpg'

    tfrecord_format_csv.loc[i,'width']    = 1024

    tfrecord_format_csv.loc[i,'height']   = 1024

    tfrecord_format_csv.loc[i,'class']    = 'wheat'

    tfrecord_format_csv.loc[i,'xmin']     = img_bbox['bbox'][i][0]

    tfrecord_format_csv.loc[i,'ymin']     = img_bbox['bbox'][i][1]

    tfrecord_format_csv.loc[i,'xmax']     = img_bbox['bbox'][i][0] + img_bbox['bbox'][i][2]

    tfrecord_format_csv.loc[i,'ymax']     = img_bbox['bbox'][i][1] + img_bbox['bbox'][i][3]

    
tfrecord_format_csv.head()
temp_df = tfrecord_format_csv[tfrecord_format_csv['filename']=='00333207f.jpg'].reset_index(drop=True)

temp_img = cv2.imread(train_images+'00333207f.jpg')

rgb_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB)

for i in range(len(temp_df)):

    rec = cv2.rectangle(rgb_img, (temp_df.loc[i,'xmin'],temp_df.loc[i,'ymin']), (temp_df.loc[i,'xmax'],temp_df.loc[i,'ymax']), (255,0,0), 2, 1) 

plt.figure(figsize=(8,8))    

plt.imshow(rec)

plt.axis('off')

plt.show()

    

tfrecord_format_csv.to_csv('tf_format_training.csv',index=False)