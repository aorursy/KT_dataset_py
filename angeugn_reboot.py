import os

import PIL

import json

import pickle

import numpy as np

from tqdm import tqdm

from IPython.display import Image, display

from PIL import ImageFont, ImageDraw



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img



input_size = (224,224)



old_cat_list = ['bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'outerwear', 'pants', 'skirts', 'tops']

new_cat_list = ['tops', 'trousers', 'outerwear', 'dresses', 'skirts']

cat_oldnew_mapping = {3:4, 7:2, 8:3, 11:1, 10:5, 9:2}



base_folder = '/kaggle'

data_folder = os.path.join( base_folder, 'input', 'til2020-test' )

train_imgs_folder = os.path.join( data_folder, 'train', 'train' )

train_annotations = os.path.join( data_folder, 'train.json' )

val_imgs_folder = os.path.join( data_folder, 'val', 'val' )

val_annotations = os.path.join( data_folder, 'val.json' )



# pickled_train = os.path.join( base_folder, 'working', 'train.p' )

pickled_val = os.path.join( base_folder, 'working', 'val_reboot.p' )
# Get all filtered image ids

all_filtered = [item for item in os.listdir( data_folder ) if item.endswith('-filtered.json')]

val_filtered = [item for item  in all_filtered if item.startswith('val')]

train_filtered = [item for item  in all_filtered if item.startswith('train')]

print(len(val_filtered))

print(len(train_filtered))

print(len(all_filtered))
def compile_ids_tolist(filtered):

  img_id_list = []

  for _json in filtered:

    json_fp = os.path.join( data_folder, _json )

    with open(json_fp, 'r') as f:

        img_ids = json.load(f)

        img_id_list.extend( img_ids )

  return img_id_list
val_img_id_list = compile_ids_tolist(val_filtered)

train_img_id_list = compile_ids_tolist(train_filtered)
with open(val_annotations, 'r') as f:

  val_gt_json = json.load(f)

with open(train_annotations, 'r') as f:

  train_gt_json = json.load(f)

with open(os.path.join(data_folder, 'train_agnes.json'), 'r') as f:

  ref_gt_json = json.load(f)

  ref_categories = ref_gt_json['categories']

  print(ref_categories)
# Computes the intersection-over-union (IoU) of two bounding boxes

def iou(bb1, bb2):

  x1,y1,w1,h1 = bb1

  xmin1 = x1 - w1/2

  xmax1 = x1 + w1/2

  ymin1 = y1 - h1/2

  ymax1 = y1 + h1/2



  x2,y2,w2,h2 = bb2

  xmin2 = x2 - w2/2

  xmax2 = x2 + w2/2

  ymin2 = y2 - h2/2

  ymax2 = y2 + h2/2



  area1 = w1*h1

  area2 = w2*h2



  # Compute the boundary of the intersection

  xmin_int = max( xmin1, xmin2 )

  xmax_int = min( xmax1, xmax2 )

  ymin_int = max( ymin1, ymin2 )

  ymax_int = min( ymax1, ymax2 )

  intersection = max(xmax_int - xmin_int, 0) * max( ymax_int - ymin_int, 0 )



  # Remove the double counted region

  union = area1+area2-intersection



  return intersection / union
# To fix multiple, we introduce non-maximum suppression, or NMS for short

def nms(detections, iou_thresh=0.7):

  dets_by_class = {}

  final_result = []

  for det in detections:

    cls = det[1]

    if cls not in dets_by_class:

      dets_by_class[cls] = []

    dets_by_class[cls].append( det )

  for _, dets in dets_by_class.items():

    candidates = list(dets)

    candidates.sort( key=lambda x:x[0], reverse=True )

    while len(candidates) > 0:

      candidate = candidates.pop(0)

      _,_,_,cx,cy,cw,ch = candidate

      copy = list(candidates)

      for other in candidates:

        # Compute the IoU. If it exceeds thresh, we remove it

        _,_,_,ox,oy,ow,oh = other

        if iou( (cx,cy,cw,ch), (ox,oy,ow,oh) ) > iou_thresh:

          copy.remove(other)

      candidates = list(copy)

      final_result.append(candidate)

  return final_result
def filterimgids_remapcats_nms( gt_json, img_id_list, cat_oldnew_mapping, ref_categories, img_folder ):

  master = {'images': [], 'annotations': []}

  imgs = gt_json['images']

  anns = gt_json['annotations']

  for ann_dict in anns:

    img_id = ann_dict['image_id']

    if img_id in img_id_list:

      #1 Categories need to be remapped and pruned if no longer relevant

      old_cat_id = ann_dict['category_id']

      if old_cat_id in cat_oldnew_mapping:

        new_cat_id = cat_oldnew_mapping[old_cat_id]

        new_ann_dict = dict(ann_dict)

        new_ann_dict['category_id'] = new_cat_id

        master['annotations'].append(new_ann_dict)

  for img_dict in imgs:

    img_id = img_dict['id']

    if img_id in img_id_list:

      new_img_dict = dict(img_dict)

      master['images'].append(new_img_dict)

  master['categories'] = ref_categories

  #2 nms needs to be applied here

  ann_acc = {}

  # Group by img_id

  for ann_dict in master['annotations']:

    img_id = ann_dict['image_id']

    if img_id not in ann_acc:

      ann_acc[img_id] = []

    ann_acc[img_id].append( ann_dict )



  # Decide anns to keep

  passed = []

  for img_id, anns in ann_acc.items():

    img_fp = os.path.join(img_folder, '{}.jpg'.format(img_id))

    W,H = PIL.Image.open(img_fp).size

    detections = []

    for ann in anns:

      left, top, width, height = ann['bbox']

      cenx = left + width/2.

      ceny = top + height/2.

      x = cenx / W

      y = ceny / H

      w = width / W

      h = height / H

      detections.append( (ann['area'], ann['category_id'], ann['id'], x,y,w,h) )

    passed_dets = nms(detections)

    passed.extend([det[2] for det in passed_dets])

  # Eliminate them

  master['annotations'] = [ann for ann in master['annotations'] if ann['id'] in passed]



  return master
val_master = filterimgids_remapcats_nms( val_gt_json, val_img_id_list, cat_oldnew_mapping, ref_categories, val_imgs_folder )

train_master = filterimgids_remapcats_nms( train_gt_json, train_img_id_list, cat_oldnew_mapping, ref_categories, train_imgs_folder )
with open('val_reboot.json', 'w') as f:

  json.dump( val_master, f )

with open('train_reboot.json', 'w') as f:

  json.dump( train_master, f )