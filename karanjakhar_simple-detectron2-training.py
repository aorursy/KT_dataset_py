import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
train_df = pd.read_csv('../input/global-wheat-detection/train.csv')
train_df.head()
train_df['width'].unique()

train_df['height'].unique()
for g in train_df.groupby('image_id'):

    b = g[1]['bbox'].values

    print(type(b),b)

    break
total_data = []

for g in tqdm_notebook(train_df.groupby('image_id')):

    data = {}

    data['filename'] = g[0]

    data['bbox'] = g[1]['bbox'].values

    total_data.append(data)
len(total_data)
!pip install cython pyyaml==5.1

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

!gcc --version
# install detectron2:

!pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()





import numpy as np

import cv2

import random





from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog


import os

import numpy as np

import json

from detectron2.structures import BoxMode



def get_wheat_dicts(total_data):

    

    dataset_dicts = []

    for idx, v in enumerate(total_data):

        record = {}

        

        filename = os.path.join('../input/global-wheat-detection/train/', v["filename"]+'.jpg')

        height, width = 1024,1024

        

        record["file_name"] = filename

        record["image_id"] = idx

        record["height"] = height

        record["width"] = width

      

        

        objs = []

        for b in v['bbox']:

            b = json.loads(b)

            obj = {

                'bbox': list(b),

                'bbox_mode': BoxMode.XYWH_ABS,

                'category_id':0,

            }

            objs.append(obj)

            

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts



from detectron2.data import DatasetCatalog, MetadataCatalog



index = int(0.9 * len(total_data))

train_data = total_data[:index]

val_data = total_data[index:]



folders = ['train', 'val']

for i, d in enumerate([train_data,val_data]):

    DatasetCatalog.register("wheat_" + folders[i], lambda d=d: get_wheat_dicts(d))

    MetadataCatalog.get("wheat_" + folders[i]).set(thing_classes=["wheat"])

wheat_metadata = MetadataCatalog.get("wheat_train")
train_data[0]['bbox'][0]
dataset_dicts = get_wheat_dicts(train_data)

for d in random.sample(dataset_dicts, 3):

    img = cv2.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=wheat_metadata, scale=1)

    vis = visualizer.draw_dataset_dict(d)

    plt.imshow(vis.get_image()[:,:,::-1])

    plt.show()
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("wheat_train",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  

cfg.SOLVER.IMS_PER_BATCH = 8

cfg.SOLVER.BASE_LR = 0.0003  

cfg.SOLVER.MAX_ITER = 4000   

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   

cfg.DATASETS.TEST = ("wheat_val", )

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode

dataset_dicts = get_wheat_dicts(val_data)

for d in random.sample(dataset_dicts, 3):    

    im = cv2.imread(d["file_name"])

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],

                   metadata=wheat_metadata, 

                   scale=0.8, 

    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.imshow(v.get_image()[:,:,::-1])

    plt.show()