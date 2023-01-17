!pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 

!pip install cython pyyaml==5.1

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

import torch, torchvision

torch.__version__

!gcc --version
# install detectron2:

!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html
import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()



# import some common libraries

import numpy as np

import cv2

import matplotlib.pyplot as plt



# import some common detectron2 utilities

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog
!ls microcontroller-detection
import pandas as pd



df = pd.read_csv('../input/microcontroller-detection/Microcontroller Detection/train_labels.csv')



df.head()
import os

import numpy as np

import json

from detectron2.structures import BoxMode

import itertools

import cv2



# write a function that loads the dataset into detectron2's standard format

def get_microcontroller_dicts(csv_file, img_dir):

    df = pd.read_csv(csv_file)

    df['filename'] = df['filename'].map(lambda x: img_dir+x)



    classes = ['Raspberry_Pi_3', 'Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora']



    df['class_int'] = df['class'].map(lambda x: classes.index(x))



    dataset_dicts = []

    for filename in df['filename'].unique().tolist():

        record = {}

        

        height, width = cv2.imread(filename).shape[:2]

        

        record["file_name"] = filename

        record["height"] = height

        record["width"] = width



        objs = []

        for index, row in df[(df['filename']==filename)].iterrows():

          obj= {

              'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],

              'bbox_mode': BoxMode.XYXY_ABS,

              'category_id': row['class_int'],

              "iscrowd": 0

          }

          objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog



classes = ['Raspberry_Pi_3', 'Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora']



for d in ["train", "test"]:

    DatasetCatalog.register('microcontroller/' + d, lambda d=d: get_microcontroller_dicts('../input/microcontroller-detection/Microcontroller Detection/' + d + '_labels.csv', '../input/microcontroller-detection/Microcontroller Detection/' + d+'/'))

    MetadataCatalog.get('microcontroller/' + d).set(thing_classes=classes)

microcontroller_metadata = MetadataCatalog.get('microcontroller/train')
import random

from detectron2.utils.visualizer import Visualizer



dataset_dicts = DatasetCatalog.get('microcontroller/train')

for d in random.sample(dataset_dicts, 10):

    img = cv2.imread(d["file_name"])

    v = Visualizer(img[:, :, ::-1], metadata=microcontroller_metadata, scale=0.5)

    v = v.draw_dataset_dict(d)

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ('microcontroller/train',)

cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.MAX_ITER = 1000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

cfg.DATASETS.TEST = ('microcontroller/test', )

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode

import random



dataset_dicts = DatasetCatalog.get('microcontroller/test')

for d in random.sample(dataset_dicts, 5):    

    im = cv2.imread(d["file_name"])

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], metadata=microcontroller_metadata, scale=0.8)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()