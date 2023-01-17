# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install tensorflow-gpu==1.15
import tensorflow as tf
print(tf.__version__)#1.15.0
import keras
print(keras.__version__)#2.3.1
keras.backend.tensorflow_backend._get_available_gpus()
tf.config.experimental.list_physical_devices()
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline
import cv2
import time
from collections import Counter
!python --version
!git clone https://github.com/matterport/Mask_RCNN.git
!cd Mask_RCNN ; python setup.py install
!pip show mask-rcnn
from os import sys
sys.path.append('./Mask_RCNN/')
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
train_csv=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/train.csv"))
df=train_csv.copy(deep=True)
df.head(),df.shape
submission=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/submission.csv"))
submission.head()
bbox=[]
for i in range(len(train_csv)):
    arr=[]
    for j in df.iloc[i][['x1','x2','y1','y2']]:
        arr.append(j)
    bbox.append(arr)
df["bbox"]=bbox
df.head()
train_csv.head()
def get_boxes(id):
    boxes=[]
    c_names=[]
    dt=df[df['name']==str(id)]
    for i in range(len(dt)):
        boxes.append(dt.iloc[i]['bbox'])
        c_names.append(dt.iloc[i]['classname'])
    return boxes,c_names
# print(get_boxes('1812.jpg'))
boxes,c_names=get_boxes('1812.jpg')
print(boxes)
print(c_names)
ctr=Counter(df['classname'])
print(len(ctr),ctr)
ct=list(ctr)
ct
SUB_DATA=list(set(submission['name']))
len(list(SUB_DATA)),SUB_DATA[:10]
ct.insert(0,'BG')
ct
class wobotDataset(Dataset):
    
    def load_dataset_2(self):
        for i in range(len(SUB_DATA)):
            di=SUB_DATA[i]
            self.add_image(
                'wobot',
                image_id=di,
                image_id_df=di,
                path=os.path.join(images,di)
            )

    def load_dataset(self,train_bool):
        D_DATA=[]
        if(train_bool):
            D_DATA=train_DATA
        else:
            D_DATA=test_DATA
            
        for i in range(len(ct)):
            self.add_class("wobot", i+1, ct[i])

        for i in range(len(D_DATA)):
            di=D_DATA[i]
            self.add_image(
                'wobot',
                image_id=di[0],
                image_id_df=di[0],
                path=os.path.join(images,di[0]),
                class_names=di[-1],
                height=di[1],
                width=di[2]
                          )
            
    def get_boxes(self,id):
        boxes=[]
        c_names=[]
        dt=df[df['name']==str(id)]
        for i in range(len(dt)):
            boxes.append(dt.iloc[i]['bbox'])
            c_names.append(dt.iloc[i]['classname'])
        return boxes,c_names
 
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        w=info['width']
        h=info['height']
        image_id_df=info['image_id_df']
        boxes,c_names= self.get_boxes(image_id_df)
        masks = np.zeros((h, w, len(boxes)), dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
#             print('i->'+str(i))
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(c_names[i]))
        return masks, np.asarray(class_ids, dtype='int32')
 
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# class wobotConfig(Config):
#     NAME = 'wobot_cfg'
#     NUM_CLASSES = 1 + 20
#     IMAGES_PER_GPU = 1
#     STEPS_PER_EPOCH = len(train_set.image_ids)
 
# config = wobotConfig()

# from mrcnn import
class MaskRCNN(MaskRCNN):
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)
        
        self.keras_model.metrics_tensors = []
    
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class PredictionConfig(Config):
    NAME = 'wobot_cfg'
    NUM_CLASSES = 1 + 20
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#https://drive.google.com/file/d/1XtgbixT2EKzteuiT2uLuzdWFgJt_Nz6g/view?usp=sharing
!wget "https://drive.google.com/uc?export=download&id=1XtgbixT2EKzteuiT2uLuzdWFgJt_Nz6g"
!wget 'https://drive.google.com/file/d/1XtgbixT2EKzteuiT2uLuzdWFgJt_Nz6g/view?usp=sharing'
!ls
!curl -L -o mask_rcnn_wobot_cfg_0002.h5 "https://drive.google.com/uc?export=download&id=1XtgbixT2EKzteuiT2uLuzdWFgJt_Nz6g"
!ls
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('../input/maskrcnn-model/mask_rcnn_wobot_cfg_0002.h5', by_name=True)
images=os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images")
print(len(os.listdir(images)))
submission_set = wobotDataset()
submission_set.load_dataset_2()
submission_set.prepare()
print('submission: %d' % len(submission_set.image_ids))
# count_sub=0
# for image_id in submission_set.image_ids:
#     count_sub+=1
#     info = submission_set.image_info[image_id]
#     print(info)
# print(count_sub)
sub_df=pd.DataFrame(columns=['name','x1','x2','y1','y2','classname'])
printer=0
for image_id in submission_set.image_ids:
#     ####
#     if image_id>5:
#         break
#     ####
    
    image = submission_set.load_image(image_id)
    name = submission_set.image_info[image_id]['image_id_df']
    
    scaled_image = mold_image(image, cfg)
    sample = np.expand_dims(scaled_image, 0)
    yhat = model.detect(sample, verbose=0)
#     x2,x1,y2,y1
    r=yhat[0]
    subb=r['rois']
    subc=r['class_ids']
    subc2=[ct[x] for x in subc]
    subb,subc,subc2
    
    for (i,j) in zip(subb,subc2):
        x2,x1,y2,y1=i
        classname=j
        sub_df.loc[len(sub_df)]=(name,x1,x2,y1,y2,classname)
        
    if(printer%100==0):
        print(printer)
    printer+=1
    
print(sub_df.shape)
# printer
sub_df
from mrcnn import visualize
image = submission_set.load_image(1673)
scaled_image = mold_image(image, cfg)
sample = np.expand_dims(scaled_image, 0)
yhat = model.detect(sample, verbose=1)
r = yhat[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ct, r['scores'])

image = submission_set.load_image(1674)
scaled_image = mold_image(image, cfg)
sample = np.expand_dims(scaled_image, 0)
yhat = model.detect(sample, verbose=1)

from mrcnn import visualize

r = yhat[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ct, r['scores'])
submit_csv=sub_df.sort_values('name',ascending=False)#,inplace=True
print(submit_csv.shape)
submit_csv
submit_csv.to_csv('submission.csv',index=False)
submit_csv=pd.read_csv('submission.csv')
submit_csv
