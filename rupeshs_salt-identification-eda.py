# linear algebra
import numpy as np 
# data processing
import pandas as pd 

depth_df=pd.read_csv("../input/depths.csv")
depth_df.info()
depth_df.head()
depth_df.hist()
depth_df[depth_df['z'].isnull()]['z'].count()
import os
from tqdm import tqdm
import hashlib

TRAIN_IMAGE_DIR = '../input/train/images/'
TRAIN_MASK_DIR = '../input/train/masks/'

train_file_lst = os.listdir(TRAIN_IMAGE_DIR)
train_mask_lst = os.listdir(TRAIN_MASK_DIR)

md5sum=[]
for file_name in tqdm(train_file_lst):
    filePath=TRAIN_IMAGE_DIR + file_name
    image_file = open(filePath, 'rb').read()
    md5sum.append(hashlib.md5(image_file).hexdigest())
train_md5sum_df = pd.DataFrame(np.column_stack([train_file_lst, md5sum]), 
                               columns=['file', 'md5sum'])
train_md5sum_df.info()
train_md5sum_df.head()
train_md5sum_df[train_md5sum_df['md5sum'].duplicated()]
from PIL import Image
img = Image.open(TRAIN_IMAGE_DIR+"b552fb0d9d.png")
img
#Mask for image "b552fb0d9d.png"

mask = Image.open(TRAIN_MASK_DIR+"b552fb0d9d.png")
mask
def read_image(file_name):    
    path = TRAIN_IMAGE_DIR+file_name
    img = Image.open(path)
    img = img.convert('RGB')
    return img
    
def read_mask(file_name):
    path = TRAIN_MASK_DIR+file_name   
    img = Image.open(path)
    #8-bit pixels, black and white
    bk = Image.new('L', size=img.size)
    g = Image.merge('RGB', (bk, img.convert('L'), bk))
    return g

from image_dataset_viz import DatasetExporter


de = DatasetExporter(read_image, read_mask, blend_alpha=0.2, n_cols=20, max_output_img_size=(100, 100))
de.export(train_file_lst, train_mask_lst, "train_dataset_viz")
!ls train_dataset_viz
ds_viz_image = Image.open("train_dataset_viz/dataset_part_0.png")
ds_viz_image