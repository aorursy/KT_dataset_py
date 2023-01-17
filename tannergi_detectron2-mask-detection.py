# install dependencies: (use cu100 because colab is on CUDA 10.0)

!pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 

!pip install cython pyyaml==5.1

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

import torch, torchvision

torch.__version__

!gcc --version

# opencv is pre-installed on colab
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
import os

import numpy as np

import json

from detectron2.structures import BoxMode

import itertools

import cv2

import xml.etree.ElementTree as ET



def get_mask_dicts(data_dir):

    classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']

    annotation_files = os.listdir(os.path.join(data_dir,'annotations'))

    

    dataset_dicts = []

    for filename in annotation_files:

        record = {}

        root = ET.parse(os.path.join(data_dir, 'annotations', filename)).getroot()

                

        record["file_name"] = os.path.join(data_dir, 'images', root.find('filename').text)

        record["height"] = int(root.find('size/height').text)

        record["width"] = int(root.find('size/width').text)

        

        objs = []

        

        for member in root.findall('object'):

            obj = {

                'bbox': [int(member[5][0].text), int(member[5][1].text), int(member[5][2].text), int(member[5][3].text)],

                'bbox_mode': BoxMode.XYXY_ABS,

                'category_id': classes.index(member[0].text),

                'iscrowd': 0

            }

            objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog



classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']



DatasetCatalog.register('mask_dataset', lambda: get_mask_dicts('../input/face-mask-detection'))

MetadataCatalog.get('mask_dataset').set(thing_classes=classes)

chess_metadata = MetadataCatalog.get('mask_dataset')
import random

from detectron2.utils.visualizer import Visualizer



dataset_dicts = DatasetCatalog.get('mask_dataset')

for d in random.sample(dataset_dicts, 10):

    img = cv2.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=chess_metadata)

    vis = visualizer.draw_dataset_dict(d)

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ('mask_dataset',)

cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 4

cfg.SOLVER.MAX_ITER = 1000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model

cfg.DATASETS.TEST = ("mask_dataset", )

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode

import random



dataset_dicts = DatasetCatalog.get('mask_dataset')

for d in random.sample(dataset_dicts, 5):    

    im = cv2.imread(d["file_name"])

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], metadata=chess_metadata, scale=0.8)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()