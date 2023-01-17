!pip install pycocotools
import json
import os
import datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import imageio as io

import pycocotools
from pycocotools.mask import encode
import pycocotools.coco as coco
from pycocotools.coco import COCO
def mask_to_bbox_corners(mask, mode='numpy'):
  '''given a binary mask (0 or int>0) returns the
     bounding box as tuple row0, row1, col0, col1 
     if mode=='XYXY' return a list row0, col0, row1,col1

     Enum of different ways to represent a box.

In detectron2:
XYXY_ABS= 0
(x0, y0, x1, y1) in absolute floating points coordinates.
The coordinates in range [0, width or height].
  '''
  col_0 = np.nonzero(mask1.any(axis=0))[0][0]
  col_1 = np.nonzero(mask1.any(axis=0))[0][-1]
  row_0 = np.nonzero(mask1.any(axis=1))[0][0]
  row_1 = np.nonzero(mask1.any(axis=1))[0][-1]
  if mode == 'numpy':
     return row_0, row_1,col_0,  col_1
  if mode == 'XYXY':
    X0 = int(col_0)
    X1 = int(col_1)
    Y0 = int(row_0)
    Y1 = int(row_1)
    return [X0, Y0, X1, Y1]

 

def groundtruth_to_masks(groundtruth):
  grey = data[N,:,:,0]
  labels = data[N,:,:,1]
  mask1 = labels == 1
  mask2 = labels == 2
  mask3 = labels == 3

  mask1 = 1* np.logical_or(mask1, mask3)
  mask2 = 1 * np.logical_or(mask2, mask3)
  return mask1.astype('uint8'), mask2.astype('uint8')
%pwd
root_dir = '/kaggle/working'
import urllib.request
url = "https://github.com/jeanpat/DeepFISH/blob/master/dataset/Cleaned_FullRes_2164_overlapping_pairs.npz?raw=true"#"https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tgz"
print ("download start!")
filename, headers = urllib.request.urlretrieve(url, filename=root_dir +"/Data2164Chrom.npz")
print ("download complete!")
print ("download file location: ", filename)
print ("download headers: ", headers)
dataset = np.load(root_dir+'/Data2164Chrom.npz')
data = dataset.f.arr_0
print(data.shape)
N = 130
grey = data[N,:,:,0]
mask1, mask2 = groundtruth_to_masks(data[N,:,:,1])

plt.figure(figsize=(10,10))
plt.subplot(141, xticks=[], yticks=[])
plt.imshow(grey, cmap=plt.cm.gray)

plt.subplot(142, xticks=[], yticks=[])
plt.title("Grountruth")
plt.imshow(data[N,:,:,1], interpolation="nearest", cmap=plt.cm.flag_r)

plt.subplot(143, xticks=[], yticks=[])
plt.title("mask:Instance 1")
plt.imshow(mask1, interpolation="nearest",cmap=plt.cm.flag_r)

plt.subplot(144, xticks=[], yticks=[])
plt.title("mask:Instance 2")
plt.imshow(mask2, interpolation="nearest",cmap=plt.cm.flag_r)

#mask2[mask2 == True]=1
print(mask2.max(), mask2.dtype)
if not(os.path.exists(root_dir+'/Data')):
  os.mkdir(root_dir+'/Data')
if os.path.isdir(root_dir+'/Data'):
  if not(os.path.exists(root_dir+'/Data/train')):
    os.mkdir(root_dir+'/Data/train')
if not(os.path.exists(root_dir+'/Data/train/annotations')):
  os.mkdir(root_dir+'/Data/train/annotations')

if not(os.path.exists(root_dir+'/Data/train/shapes_train2020')):
  os.mkdir(root_dir+'/Data/train/shapes_train2020')
%ls /kaggle/working
%ls /kaggle/working/Data
%ls /kaggle/working/Data/train
%ls /kaggle/working/Data/train/annotations
%ls /kaggle/working/Data/train/shapes_train2020
labels = {1:'chromosome'}
train_path = os.path.join(root_dir,'Data','train')
subset ='shapes_train'+'2020'
annotations = 'annotations'
N = 130
image_id = "{:04d}".format(N)
print(image_id)
grey_file_name = os.path.join(image_id+'.png')
path_to_grey = os.path.join(train_path,subset, grey_file_name)

mask1_file_name = image_id+'_'+ labels[1]+'_'+str(1)+'.png'
path_to_mask1 = os.path.join(train_path, annotations, mask1_file_name)

mask2_file_name = image_id+'_'+ labels[1]+'_'+str(2)+'.png'
path_to_mask2 = os.path.join(train_path, annotations, mask2_file_name)
print(path_to_mask1)
print(path_to_mask2)

mask1 = mask1.astype(np.uint8)
mask2 = mask2.astype(np.uint8)

print(os.path.exists(os.path.join(train_path, subset)))
print(os.path.join(train_path, subset, grey_file_name))#/Data\ Science/dataset

io.imsave(os.path.join(train_path, subset, grey_file_name),grey)
io.imsave(os.path.join(train_path, annotations, mask1_file_name), mask1)
io.imsave(os.path.join(train_path, annotations, mask2_file_name), mask2)

%ls /kaggle/working
%ls /kaggle/working/Data
%ls /kaggle/working/Data/train
%ls /kaggle/working/Data/train/annotations
%ls /kaggle/working/Data/train/shapes_train2020


#%ls My\ Drive/Science/Data\ Science/dataset/shapes/train/annotations
N = 130
NUM_CATEGORIES = 2 # chrom:1, background :0
grey = data[N,:,:,0]
# dictionnary for image 130
## path to greyscaled image : /kaggle/working/Data/train
dataset_root = os.path.join('/kaggle/working/Data/train')
subset ='shapes_train'+'2020'
#annotations = 'annotations'
image_id = "{:04d}".format(N) #Possible bug here since
#print(image_id)
grey_file_name = os.path.join(image_id+'.png')
path_to_grey = os.path.join(dataset_root,subset, grey_file_name)

dict_to_130 = {}
dict_to_130['file_name']= path_to_grey

## grey shape
dict_to_130['height']= grey.shape[0]
dict_to_130['width']= grey.shape[1]

## the image id could be different from its index, here choose id=index=N
dict_to_130['image_id'] = "{:04d}".format(N)#N

### Prepare the dicts for annotation
#### bounding boxes : theres two instances in image 130:First instance
dict_to_130['annotations']= []
annotation_instance_01_dict = {}
annotation_instance_01_dict['bbox']=None
Bbox_0130_01 = mask_to_bbox_corners(mask1, mode='XYXY')

print("     ", type(Bbox_0130_01), type(Bbox_0130_01[0]))

annotation_instance_01_dict['bbox'] = Bbox_0130_01
annotation_instance_01_dict['bbox_mode']=0 #XYXY
annotation_instance_01_dict['category_id'] = NUM_CATEGORIES-1

annotation_instance_01_dict['segmentation']=None # A dict is used, How to handle several instances?
mask1 = mask1 > 0
### rle_instance_1 is a dict
###
### <byte> type issue !!!
###

rle_instance_1 = encode(np.asarray(mask1, order="F"))

print("rle_instance1 ",rle_instance_1)
print("rle_instance1['counts'] is of type:",type(rle_instance_1['counts']))

print("rle_instance1 ",rle_instance_1['counts'].decode("utf-8"))

counts_byte_to_utf8 = rle_instance_1['counts'].decode("utf-8")
rle_instance_1['counts'] = counts_byte_to_utf8
###
###
#cfg.INPUT.MASK_FORMAT='bitmask'
annotation_instance_01_dict['segmentation'] = rle_instance_1

dict_to_130['annotations'].append(annotation_instance_01_dict)

#### bounding boxes : theres two instances in image 130: second instance

annotation_instance_02_dict = {}
annotation_instance_02_dict['bbox']=None
Bbox_0130_02 = mask_to_bbox_corners(mask2, mode='XYXY')
print("     ", type(Bbox_0130_02))
annotation_instance_02_dict['bbox'] = Bbox_0130_02
annotation_instance_02_dict['bbox_mode']=0 #XYXY
annotation_instance_02_dict['category_id'] = NUM_CATEGORIES-1

annotation_instance_02_dict['segmentation']=None # A dict is used, How to handle several instances?
mask2 = mask2 > 0
### rle_instance_1 is a dict
rle_instance_2 = encode(np.asarray(mask2, order="F"))
#cfg.INPUT.MASK_FORMAT='bitmask'

###
### <byte> type issue !!!
###
rle_instance_2['counts'] = rle_instance_2['counts'].decode("utf-8")
###
annotation_instance_02_dict['segmentation'] = rle_instance_2
dict_to_130['annotations'].append(annotation_instance_02_dict)

row_0, row_1, col_0, col_1 = mask_to_bbox_corners(mask1, mode='numpy')

box_mask = np.zeros(grey.shape, dtype=grey.dtype)
box_mask[row_0-1:row_1+1,col_0-1:col_1+1]=1

print(coco.maskUtils.decode(rle_instance_2))
plt.figure(figsize=(10,10))

plt.subplot(131, xticks=[], yticks=[])
plt.imshow(grey, cmap=plt.cm.gray)

plt.subplot(132, xticks=[], yticks=[])
plt.title("rle -> mask")
plt.imshow(coco.maskUtils.decode(rle_instance_2), interpolation="nearest", cmap=plt.cm.flag_r)

plt.subplot(133, xticks=[], yticks=[])
plt.title("mask + bounding box")
plt.imshow(1*mask1+5*box_mask, interpolation="nearest",cmap=plt.cm.flag_r)
print(dict_to_130.keys())
print("    ", type(dict_to_130['height']), dict_to_130['height'])
print("    ", type(dict_to_130['width']), dict_to_130['width'])
print("    ", type(dict_to_130['image_id']), dict_to_130['image_id'])
print(dict_to_130['file_name'])
print(type(dict_to_130['annotations']))

print(dict_to_130['annotations'])
print(dict_to_130['annotations'][0].keys())
print(dict_to_130['annotations'][0]['segmentation'])
print(dict_to_130['annotations'][0]['segmentation'])
print(dict_to_130['annotations'][0]['segmentation'].keys())
print("    ",dict_to_130['annotations'][0]['segmentation']['size'],"---",type(dict_to_130['annotations'][0]['segmentation']['size'][0]))
print("    ",dict_to_130['annotations'][0]['segmentation']['counts'])
print("    ",type(dict_to_130['annotations'][0]['segmentation']['counts']))
my_json = '/kaggle/working/Data/train/annotations/instances_0130_data.json'
print(my_json)

#print(os.path.exists(os.path.join('','kaggle','working','Data','train','annotations')))
print(os.path.exists('/kaggle/working/Data/train/annotations'))
%ls /kaggle/working/Data/train/annotations
%pwd
with open(my_json, 'w') as f:
    json.dump(dict_to_130, f)
#%ls /kaggle/working
#%ls /kaggle/working/Data
#%ls /kaggle/working/Data/train
%ls /kaggle/working/Data/train/annotations
#%ls /kaggle/working/Data/train/shapes_train2020
annotations_path= my_json
dataType='0130_data'
annFile=(my_json,dataType)
annFile
my_coco_dataset=COCO(my_json) # WRONG
#from detectron2.data.datasets import register_coco_instances
from detectron2.data import datasets
from detectron2.data.datasets import register_coco_instances
path_to_annotation = my_json
path_to_image = path_to_grey

#register_coco_instances("chromosome", {}, "./data/trainval.json", "./data/images"
register_coco_instances("chromosome_dataset_train", {}, path_to_annotation, path_to_image)
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
#visualize training data
my_chromosome_train_metadata = MetadataCatalog.get("chromosome_dataset_train")
dataset_dicts = DatasetCatalog.get("chromosome_dataset_train")

