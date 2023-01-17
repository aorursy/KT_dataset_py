import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
train_df=pd.read_csv("../input/train.csv")
train_df.head()
train_df.info()
train_df['labels'].isnull().sum()

#276 NAN 
import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import tqdm

from PIL import Image

%matplotlib inline



img=Image.open("../input/train_images/100241706_00004_2.jpg")

plt.imshow(img)
image_ids=train_df["image_id"].tolist()

image_size=[]

for image_id in image_ids:

    img=Image.open("../input/train_images/"+image_id+".jpg")

    width,height=img.size

    image_size.append((width,height))

   

image_size_df=pd.DataFrame(image_size,columns=["width","height"])

image_size_df.head()
names = ['width', 'height']

plt.hist([image_size_df["width"], image_size_df["height"]],label=names)

plt.legend()
trainimage_lst=train_df.head(20).image_id.tolist()

fig=plt.figure(figsize=(64, 64))

columns = 4

rows = 5

for index,image_id  in enumerate(trainimage_lst):

    img=Image.open("../input/train_images/"+image_id+".jpg")

    fig.add_subplot(rows, columns, index+1)

    plt.imshow(img)

    plt.axis('off')

plt.show()

translation_df=pd.read_csv("../input/unicode_translation.csv")
translation_df.head(10)
translation_df.info()