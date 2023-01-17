!pip install boxx > .null
import boxx  
# boxx: A Tool-box for Efficient Build and Debug in Python. Especially for Scientific Computing and Computer Vision.
# For more infomation about boxx : https://github.com/DIYer22/boxx

import numpy as np
import pandas as pd 
import random
import os
print(os.listdir("../input/retail_product_checkout"))
# visualization train image (single image)
import glob
from skimage.io import imread
train_image = imread(glob.glob("../input/retail_product_checkout/train2019/*")[0])
boxx.show(train_image)
# visualization test/val image (checkout image)
test_image = imread(glob.glob("../input/retail_product_checkout/test2019/*")[0])
boxx.show(test_image)


val_image = imread(glob.glob("../input/retail_product_checkout/val2019/*")[0])
boxx.show(val_image)
# Loading annotation files
train_js = boxx.loadjson('../input/retail_product_checkout/instances_train2019.json')
val_js = boxx.loadjson('../input/retail_product_checkout/instances_val2019.json')
test_js = boxx.loadjson('../input/retail_product_checkout/instances_test2019.json')

# Visualization struct of instances_train2019.json
# These annotation files has similar struct as COCO Object Detection Dataset
boxx.tree(train_js, deep=1)
boxx.tree(val_js, deep=1)
boxx.tree(test_js, deep=1)
# Visualization struct of instances_test2019.json
from pprint import pprint
pprint(test_js['images'][0])
pprint(test_js['annotations'][0])
# The Categories Data format
categories_df = pd.DataFrame(train_js['categories'])
categories_df
# Statistic the RPC dataset in different split set

def statistic_rpc_json_dataset(js, split_name=None):
    '''
    statistic dataset, input a coco format json file, then print and return `boxx.Markdown` instance
    note: `boxx.Markdown` is a sub class of `pd.DataFrame`
    '''
    df = pd.DataFrame(js['annotations'])
    images = len(js['images'])
    objects = len(js['annotations'])
    
    object_number_per_image = df.groupby('image_id').id.count().mean()
    category_number_per_image = df.groupby('image_id').apply(lambda sdf: len(set(sdf.category_id))).mean()
    
    markdown_df = pd.DataFrame([dict(split_name=split_name, 
                                     images=images, objects=objects, 
                                     object_number_per_image=round(object_number_per_image,2), 
                                     category_number_per_image=round(category_number_per_image,2))])
    markdown = boxx.Markdown(markdown_df[['split_name', 'images', 'objects',  'object_number_per_image','category_number_per_image', ]])
    #boxx.g()
    print(markdown)
    return markdown

statistic_rpc_json_dataset(train_js, 'train')
statistic_rpc_json_dataset(val_js, 'val')
statistic_rpc_json_dataset(test_js, 'test')

# Statistic checkout(val+test) set

checkout_js = dict(images=test_js['images']+val_js['images'], annotations=test_js['annotations']+val_js['annotations'])
statistic_rpc_json_dataset(checkout_js, 'checkout(val+test)')
# Statistic checkout(val+test) sets for different clutters

for level in ["easy", "medium", "hard"]:
    level_images = filter(lambda d:d['level']==level, test_js['images']+val_js['images'])
    level_images = list(level_images)
    
    level_image_ids = set([d['id'] for d in level_images])
    level_annotations = list(filter(lambda d:d['image_id'] in level_image_ids, test_js['annotations']+val_js['annotations'] ))
    
    level_js = dict(images=level_images,annotations=level_annotations)
    statistic_rpc_json_dataset(level_js, level)