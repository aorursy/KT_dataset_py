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

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog
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
from detectron2.data import detection_utils as utils

import detectron2.data.transforms as T

import copy



def custom_mapper(dataset_dict):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [

        T.Resize((800,600)),

        T.RandomBrightness(0.8, 1.8),

        T.RandomContrast(0.6, 1.3),

        T.RandomSaturation(0.8, 1.4),

        T.RandomRotation(angle=[90, 90]),

        T.RandomLighting(0.7),

        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),

    ]

    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))



    annos = [

        utils.transform_instance_annotations(obj, transforms, image.shape[:2])

        for obj in dataset_dict.pop("annotations")

        if obj.get("iscrowd", 0) == 0

    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])

    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict
from detectron2.engine import DefaultTrainer

from detectron2.data import build_detection_test_loader, build_detection_train_loader



class CustomTrainer(DefaultTrainer):

    @classmethod

    def build_train_loader(cls, cfg):

        return build_detection_train_loader(cfg, mapper=custom_mapper)
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
from detectron2.config import get_cfg



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ('microcontroller/train',)

cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 16

cfg.SOLVER.MAX_ITER = 1000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CustomTrainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
# code from https://www.kaggle.com/julienbeaulieu/detectron2-wheat-detection-eda-training-eval

train_data_loader = trainer.build_train_loader(cfg)

data_iter = iter(train_data_loader)

batch = next(data_iter)
rows, cols = 3, 3

plt.figure(figsize=(20,20))



for i, per_image in enumerate(batch[:int(rows*cols)]):

    

    plt.subplot(rows, cols, i+1)

    

    # Pytorch tensor is in (C, H, W) format

    img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()

    img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)



    visualizer = Visualizer(img, metadata=microcontroller_metadata, scale=0.5)



    target_fields = per_image["instances"].get_fields()

    labels = None

    vis = visualizer.overlay_instances(

        labels=labels,

        boxes=target_fields.get("gt_boxes", None),

        masks=target_fields.get("gt_masks", None),

        keypoints=target_fields.get("gt_keypoints", None),

    )

    plt.imshow(vis.get_image()[:, :, ::-1])
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model

cfg.DATASETS.TEST = ('microcontroller/test', )

predictor = DefaultPredictor(cfg)
df_test = pd.read_csv('../input/microcontroller-detection/Microcontroller Detection/test_labels.csv')

df_test
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