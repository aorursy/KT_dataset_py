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
import matplotlib.pyplot as plt

import cv2

from glob import glob

from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET
PATH = os.path.abspath("../input/dhakaai-dhaka-based-traffic-detection-dataset/train/Final Train Dataset")

images_jpg = glob(os.path.join(PATH,"*.jpg"))

images_JPG = glob(os.path.join(PATH,"*.JPG"))

images_jpeg = glob(os.path.join(PATH,"*.jpeg"))

images_png = glob(os.path.join(PATH,"*.png"))

images_PNG = glob(os.path.join(PATH,"*.PNG"))

images=sorted(images_jpg+images_jpeg+images_png+images_JPG+images_PNG)
xml_files=sorted(glob(os.path.join(PATH, "*.xml")))
images.remove('/kaggle/input/dhakaai-dhaka-based-traffic-detection-dataset/train/Final Train Dataset/231.jpg')

xml_files.remove('/kaggle/input/dhakaai-dhaka-based-traffic-detection-dataset/train/Final Train Dataset/231.xml')
def proc_images(images):

    """

    Returns : 

        x is an array of resized images

        

    """



    x = [] # images as arrays

    

    WIDTH = 1024

    HEIGHT = 1024



    for img in images:

       

        # Read and resize image

        full_size_image = cv2.imread(img)

        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))



    return x
x = proc_images(images)
#collecting images name

image_ids=[]

for image_id in images:

    ids=image_id.split('/')[6].split('.')[0]

    image_ids.append(ids)

image_ids[0:10]
# Set it up as a dataframe

df = pd.DataFrame()

df["image_id"]=image_ids

df["images"]=x

df.head()


xml_df=pd.DataFrame(columns=['image_id','class','xmin','ymin','xmax','ymax'])

dictionary={}

for xml_dir in xml_files:

    

    tree = ET.parse(xml_dir)

    root = tree.getroot()



    i=xml_dir.split('/')[6].split('.')[0]

    sample_annotations = []

    for objects in root.iter('object'):

        vehicle=objects.find('name').text

    

    



        for neighbor in objects.iter('bndbox'):

            xmin = int(neighbor.find('xmin').text)

            ymin = int(neighbor.find('ymin').text)

            xmax = int(neighbor.find('xmax').text)

            ymax = int(neighbor.find('ymax').text)

            

            

            if i in dictionary.keys():

                dictionary[i].append([vehicle,xmin, ymin, xmax, ymax])

                

            

            else:

                dictionary[i]=[[vehicle,xmin, ymin, xmax, ymax]]

        

        

        

        xml_df.loc[objects]=[i,vehicle,xmin, ymin, xmax, ymax]

        

                   
xml_df.head()
classes=list(xml_df['class'].unique())

classes
#Display 12 train images



row = 3; col = 4;



plt.figure(figsize=(15,int(15*row/col)))



for j,img in enumerate(df['images'].loc[0:row*col-1]):

    

    plt.subplot(row,col,j+1)

    plt.axis('off')

    plt.imshow(img)

        

plt.show()
row = 3; col = 4;



plt.figure(figsize=(15,int(15*row/col)))



for j,img in enumerate(images[0:row*col]):

    

    plt.subplot(row,col,j+1)

    plt.axis('off')

    sample_image_annotated=Image.open(img)

    

    img_bbox = ImageDraw.Draw(sample_image_annotated)

    

    keys=list(dictionary.keys())

    for bbox in dictionary[keys[j]]:

        img_bbox.rectangle(bbox[1:], outline="red",width=3)

        

    plt.imshow(sample_image_annotated)
xml_df.to_csv('train_bbox.csv',index=False)
#np.savez("train_images_arrays", df)