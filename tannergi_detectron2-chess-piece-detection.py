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





# write a function that loads the dataset into detectron2's standard format

def get_dataset_dicts(annotation_path, image_path):

    classes = ['white-rook', 'white-knight', 'white-bishop', 'white-king', 'white-queen', 'white-pawn', 'black-rook', 'black-knight', 'black-bishop', 'black-king', 'black-queen', 'black-pawn']



    dataset_dicts = []

    for filename in os.listdir(annotation_path):

        record = {}

        root = ET.parse(os.path.join(annotation_path, filename)).getroot()

        record["file_name"] = os.path.join(image_path, root.find('filename').text)

        record["height"] = 2248

        record["width"] = 4000



        objs = []

        for member in root.findall('object'):

          obj = {

              'bbox': [int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text)],

              'bbox_mode': BoxMode.XYXY_ABS,

              'category_id': classes.index(member[0].text),

              "iscrowd": 0

          }

          objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog



classes = ['white-rook', 'white-knight', 'white-bishop', 'white-king', 'white-queen', 'white-pawn', 'black-rook', 'black-knight', 'black-bishop', 'black-king', 'black-queen', 'black-pawn']



DatasetCatalog.register('chess_piece_dataset', lambda: get_dataset_dicts('../input/chess-piece-detection/Chess Detection/annotations', '../input/chess-piece-detection/Chess Detection/images'))

MetadataCatalog.get('chess_piece_dataset').set(thing_classes=classes)

chess_metadata = MetadataCatalog.get('chess_piece_dataset')
import random

from detectron2.utils.visualizer import Visualizer



dataset_dicts = DatasetCatalog.get('chess_piece_dataset')

for d in random.sample(dataset_dicts, 10):

    img = cv2.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=chess_metadata, scale=0.5)

    vis = visualizer.draw_dataset_dict(d)

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ('chess_piece_dataset',)

cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.MAX_ITER = 1000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model

cfg.DATASETS.TEST = ("chess_piece_dataset", )

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode

import random



dataset_dicts = DatasetCatalog.get('chess_piece_dataset')

for d in random.sample(dataset_dicts, 5):    

    im = cv2.imread(d["file_name"])

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], metadata=chess_metadata, scale=0.8)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()