import numpy as np

import pandas as pd

import os

import seaborn as sns

from glob import glob

from PIL import Image,ImageDraw

from matplotlib import pyplot as plt
img = Image.open('/kaggle/input/pascal-voc-2012/VOC2012/JPEGImages/2010_003253.jpg')
img
len(os.listdir('/kaggle/input/pascal-voc-2012/VOC2012/Annotations/'))
import xml.etree.ElementTree as ET
def parsexml(filename,labels):

    tree = ET.parse(filename)

    op=[]

    for obj in tree.findall('object'):

        # if not check(obj.find('name').text):

        #     continue

        global cntpos,cntneg

        obj_struct = {}

        obj_struct['name'] = obj.find('name').text

        labels.add(obj_struct['name'])

        #print(obj_struct['name'])

        bbox = obj.find('bndbox')

        obj_struct['bbox'] = [float(bbox.find('xmin').text) - 1,

                              float(bbox.find('ymin').text) - 1,

                              float(bbox.find('xmax').text) - 1,

                              float(bbox.find('ymax').text) - 1]

        xmin=float(bbox.find('xmin').text)

        ymin=float(bbox.find('ymin').text)

        xmax=float(bbox.find('xmax').text)

        ymax=float(bbox.find('ymax').text)

        op+=[[xmax-xmin,ymax-ymin]]

    return op
filesizes = []

labels=set()

for file in glob('/kaggle/input/pascal-voc-2012/VOC2012/Annotations/*xml'):

    filesizes+=parsexml(file,labels)
labels
len(filesizes)
filesizes[0:3][:]
sizes = np.array(filesizes)
sizes.mean(axis=0)
np.mean(sizes[:,0]/sizes[:,1])
sns.distplot(sizes[:,1])
ratio = sizes[:,0]/sizes[:,1]

9
np.median(ratio)
sns.distplot(ratio,bins=50)
ratio_range = np.linspace(0.2,5,25)

ratio_count = np.zeros(25)
for r in ratio:

    ratio_count[min(int(r//0.2),24)]+=1
ratio_count
df1=pd.DataFrame()

df1['ratio']=ratio_range

df1['count']=ratio_count

df1.to_csv('ratio_count.csv',index=False)

df1
print(np.sum(ratio_count[0:3]))

print(np.sum(ratio_count[3:6]))

print(np.sum(ratio_count[6:]))

width_range = np.linspace(64,512,8)

width_range.shape
width_range
size_counts=np.zeros((8,8))
for x,y in sizes:

    size_counts[int(x//64),int(y//64)]+=1
np.max(size_counts)
size_counts
df = pd.DataFrame()

df['width(rows)\height(cols)']=width_range

for i in range(8):

    df[width_range[i]]=size_counts[:,i]
df
df.to_csv('size_counts.csv',index=False)
def draw_bbox(img,xmin,ymin,xmax,ymax):

    draw = ImageDraw.Draw(img)

    fill=(0,255,0)

    draw.line(((xmin,ymin),(xmax,ymin)),fill=fill)

    draw.line(((xmin,ymin),(xmin,ymax)),fill=fill)

    draw.line(((xmax,ymin),(xmax,ymax)),fill=fill)

    draw.line(((xmin,ymax),(xmax,ymax)),fill=fill)
img=Image.open('/kaggle/input/pascal-voc-2012/VOC2012/JPEGImages/2007_000256.jpg')

draw_bbox(img,8,96,491,232)

img
def extract_objects(filename,class_count):

    tree = ET.parse(filename)

    p=tree.find('filename').text

    img = Image.open('/kaggle/input/pascal-voc-2012/VOC2012/JPEGImages/'+p).convert('RGB')

    for obj in tree.findall('object'):

        # if not check(obj.find('name').text):

        #     continue

        obj_struct = {}

        obj_struct['name'] = obj.find('name').text

        #print(obj_struct['name'])

        bbox = obj.find('bndbox')

        xmin=float(bbox.find('xmin').text)

        ymin=float(bbox.find('ymin').text)

        xmax=float(bbox.find('xmax').text)

        ymax=float(bbox.find('ymax').text)

        crop_img=img.crop((xmin,ymin,xmax,ymax))

        crop_img.save('/kaggle/working/objimages/'+obj_struct['name']+'/'+str(class_count[obj_struct['name']])+".jpg")

        class_count[obj_struct['name']]+=1

        

        
for lbl in labels:

    os.makedirs('/kaggle/working/objimages/'+lbl)
class_count={lbl:0 for lbl in labels}

for file in glob('/kaggle/input/pascal-voc-2012/VOC2012/Annotations/*xml'): 

    extract_objects(file,class_count)
class_count