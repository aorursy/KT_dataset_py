!pip install pycocotools
import numpy as np 

import pandas as pd

import os

import cv2

# visualization

import matplotlib.pyplot as plt

from matplotlib import patches as patches

# plotly offline imports

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly import subplots

import plotly.express as px

import plotly.figure_factory as ff

from plotly.graph_objs import *

from plotly.graph_objs.layout import Margin, YAxis, XAxis

init_notebook_mode()

# frequent pattern mining





from pycocotools.coco import COCO

from pycocotools.mask import encode,decode,area,toBbox



import json
data_path = '../input/understanding_cloud_organization'

train_csv_path = os.path.join(data_path,'train.csv')

train_image_path = os.path.join(data_path,'train_images')

pd.read_csv(train_csv_path).head()
train_df = pd.read_csv(train_csv_path).fillna(-1)

train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

# lets create a dict with class id and encoded pixels and group all the defaults per image

train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)
cats_dic={'background':0,'Fish':1, 'Flower':2, 'Gravel':3, 'Sugar':4}
grouped_EncodedPixels = train_df.groupby('ImageId')['Label_EncodedPixels'].apply(list)
def np_resize(img, input_shape):

    """

    Reshape a numpy array, which is input_shape=(height, width), 

    as opposed to input_shape=(width, height) for cv2

    """

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(rle, input_shape=(1400,2100)):

    width, height = input_shape[:2]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return mask.reshape(height, width).T



def build_masks(rles, input_shape, reshape=None):

    depth = len(rles)

    if reshape is None:

        masks = np.zeros((*input_shape, depth))

    else:

        masks = np.zeros((*reshape, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            if reshape is None:

                masks[:, :, i] = rle2mask(rle, input_shape)

            else:

                mask = rle2mask(rle, input_shape)

                reshaped_mask = np_resize(mask, reshape)

                masks[:, :, i] = reshaped_mask

    

    return masks



def build_rles(masks, reshape=None):

    width, height, depth = masks.shape

    

    rles = []

    

    for i in range(depth):

        mask = masks[:, :, i]

        

        if reshape:

            mask = mask.astype(np.float32)

            mask = np_resize(mask, reshape).astype(np.int64)

        

        rle = mask2rle(mask)

        rles.append(rle)

        

    return rles
def mask2polygon(mask):

    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []

    for contour in contours:

        contour_list = contour.flatten().tolist()

        if len(contour_list) > 4:# and cv2.contourArea(contour)>10000

            segmentation.append(contour_list)

    return segmentation



def rlestr2list(rlestr):

    array = np.asarray([int(x) for x in rlestr.split()])

    return array

def rlestr2rleseg(rlestr):

    segmentation={"counts":rlestr2list(rlestr), "size": [1400, 2100]}

    return segmentation
def mask2area(mask):

    return area(encode(mask))
def bounding_box(img):

    # return max and min of a mask to draw bounding box

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]



    return rmin, rmax, cmin, cmax
class NpEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.integer):

            return int(obj)

        elif isinstance(obj, np.floating):

            return float(obj)

        elif isinstance(obj, np.ndarray):

            return obj.tolist()

        else:

            return super(NpEncoder, self).default(obj)
def convert(grouped_EncodedPixels,categories, json_file="test.json"):

    """

    json_file : 保存生成的json文件路径

    """

    json_dict = {"images": [], "type": "instances", "annotations": [],

                 "categories": []}

    bnd_id = 1

    image_id=0

    

    for img_name in grouped_EncodedPixels.index:

        image_id +=1

        height, width=1400,2100

        image = {'file_name': img_name, 'height': height, 'width': width,

                 'id': image_id}

#         print(image)

        json_dict['images'].append(image)

        rle_lists=grouped_EncodedPixels[img_name]

        for rle in rle_lists:

            # 可能需要根据具体格式修改的地方

            category = rle[0]

            if category not in categories:

                new_id = len(categories)

                categories[category] = new_id

            category_id = categories[category]

            rlestr=rle[1]

            if rlestr!=-1:

#                 print(category)

                mask=rle2mask(rlestr)

                ymin,ymax,xmin,xmax=bounding_box(mask)

            

#                 print(xmin, ymin, xmax, ymax)

                assert(xmax > xmin)

                assert(ymax > ymin)

                o_width = abs(xmax - xmin)

                o_height = abs(ymax - ymin)

                ann = {'area': o_width*o_height, 'iscrowd':0, 'image_id':

                       image_id, 'bbox': [xmin, ymin, o_width, o_height],

                       'category_id': category_id, 'id': bnd_id, 'ignore': 0,

                       'segmentation':mask2polygon(mask)}

                json_dict['annotations'].append(ann)

                bnd_id = bnd_id + 1

    for cate, cid in categories.items():

        cat = {'supercategory': 'none', 'id': cid, 'name': cate}

        json_dict['categories'].append(cat)

#     print(json_dict)

    json_fp = open(json_file, 'w',encoding='utf-8')

    json_str = json.dumps(json_dict,cls=NpEncoder)

    json_fp.write(json_str)

    json_fp.close()
# convert(grouped_EncodedPixels,cats_dic)
!ls
%matplotlib inline

from pycocotools.coco import COCO

from pycocotools.mask import encode,decode,area,toBbox



import numpy as np

import skimage.io as io

import matplotlib.pyplot as plt

import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)





annFile='test.json'

def test():

    coco=COCO(annFile)



    imgIds = coco.getImgIds()

    imags=coco.loadImgs(imgIds)



    annIds = coco.getAnnIds(imgIds=imgIds)

    ann = coco.loadAnns(annIds)[0]



    mask=coco.annToMask(ann)

    rle=coco.annToRLE(ann)



    rle=encode(mask)

    mask=decode(rle)



    area(rle)

    toBbox(rle)
# test()
import xml.etree.ElementTree as ET
xmlstr='''<annotation>

	<folder>VOC2007</folder>

	<filename>000001.jpg</filename>

	<source>

		<database>The VOC2007 Database</database>

		<annotation>PASCAL VOC2007</annotation>

		<image>flickr</image>

		<flickrid>341012865</flickrid>

	</source>

	<owner>

		<flickrid>Fried Camels</flickrid>

		<name>Jinky the Fruit Bat</name>

	</owner>

	<size>

		<width>2100</width>

		<height>1400</height>

		<depth>3</depth>

	</size>

	<segmented>0</segmented>

	

</annotation>

'''

objectstr='''

    <object>

		<name>person</name>

		<pose>Left</pose>

		<truncated>1</truncated>

		<difficult>0</difficult>

		<bndbox>

			<xmin>8</xmin>

			<ymin>12</ymin>

			<xmax>352</xmax>

			<ymax>498</ymax>

		</bndbox>

	</object>

    '''
def convert2voc(grouped_EncodedPixels,categories,base_dir="."):



    json_dict = {"images": [], "type": "instances", "annotations": [],

                 "categories": []}

    bnd_id = 1

    image_id=0

    

    for img_name in grouped_EncodedPixels.index:

        root = ET.fromstring(xmlstr)

        root.find('filename').text=img_name

        rle_lists=grouped_EncodedPixels[img_name]

        

        for rle in rle_lists:

            rlestr=rle[1]

            if rlestr!=-1:

                object_el=ET.fromstring(objectstr)

                object_el.find('name').text=rle[0]

                mask=rle2mask(rlestr)

                ymin,ymax,xmin,xmax=bounding_box(mask)

                assert(xmax > xmin)

                assert(ymax > ymin)

                bndbox=object_el.find('bndbox')

                bndbox.find('ymin').text=str(ymin)

                bndbox.find('ymax').text=str(ymax)

                bndbox.find('xmin').text=str(xmin)

                bndbox.find('xmax').text=str(xmax)

                root.append(object_el)

        rough_string = ET.tostring(root, encoding="utf-8", method="xml")

        filename=img_name.replace("jpg","xml")

        filename=os.path.join(base_dir,"Annotations",filename)

        with open(filename,'wb') as f:

            f.write(rough_string)


!mkdir Annotations

!mkdir ImageSets



!mkdir ImageSets/Main

!ls
base_dir="."
# convert2voc(grouped_EncodedPixels,cats_dic)
!ls Annotations
def create_dataset(grouped_EncodedPixels,base_dir=".",filename="train.txt"):

    filename=os.path.join(base_dir,"ImageSets/Main",filename)

    with open(filename,'w') as f:

        for img_name in grouped_EncodedPixels.index:

            img_name=img_name.replace(".jpg","")

            f.writelines(img_name)

            f.writelines("\n")
# create_dataset(grouped_EncodedPixels,base_dir)
!ls ImageSets/Main