# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

from sklearn import preprocessing

import itertools

import matplotlib.pyplot as plt

import pickle

import copy



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls ../input/detectron2/
# # install dependencies: (use cu101 because colab has CUDA 10.1)

# !pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 

# !pip install cython pyyaml==5.1

# !pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# import torch, torchvision

# print(torch.__version__, torch.cuda.is_available())

# !gcc --version

# # opencv is pre-installed on colab
# !pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html

!pip install ../input/detectron2/pycocotools-2.0.0-cp37-cp37m-linux_x86_64.whl

!pip install ../input/detectron"2"/torch-1.5.0cu101-cp37-cp37m-linux_x86_64.whl

!pip install ../input/detectron2/torchvision-0.6.0cu101-cp37-cp37m-linux_x86_64.whl

!pip install ../input/detectron2/yacs-0.1.7-py3-none-any.whl

!pip install ../input/detectron2/fvcore-0.1.dev200505-py3-none-any.whl

!pip install ../input/detectron2/detectron2-0.1.2cu101-cp37-cp37m-linux_x86_64.whl
# You may need to restart your runtime prior to this, to let your installation take effect

# Some basic setup:

# Setup detectron2 logger

import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()



# import some common libraries

import numpy as np

import cv2

import random



# import some common detectron2 utilities

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog

from detectron2.structures import BoxMode

from ast import literal_eval

from detectron2.data import build_detection_train_loader

from detectron2.data import transforms as T

from detectron2.data import detection_utils as utils
train = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')
!ls ../working/
le = preprocessing.LabelEncoder()

le.fit(train['source'].values)
train['bbox'] = train['bbox'].apply(literal_eval)
image_ids = train['image_id'].values
grp_image_id = train.groupby('image_id')
def create_datatset():    

    img_dir = '/kaggle/input/global-wheat-detection/train/'

    dataset_dicts = []



    for img_id in image_ids:

        image_anno_df = grp_image_id.get_group(img_id)

        record = {}

        file_path = '{}{}.jpg'.format(img_dir, img_id)

        record["file_name"] = file_path

        record["image_id"] = img_id

        record["height"] = int(image_anno_df.iloc[0].height)

        record["width"] = int(image_anno_df.iloc[0].width)



        objs = []



        for _,row in image_anno_df.iterrows():

            bbox = row['bbox']

            xmin, ymin, width, height = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            xmax = xmin + width

            ymax = ymin + height



            obj = {

            "bbox": [xmin, ymin, xmax, ymax],

            "bbox_mode": BoxMode.XYXY_ABS,

            "category_id": 1,

            "iscrowd": 0

              }



            objs.append(obj)



        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts

  
def get_train_data():

    with open('../input/wheat-datastet/wheat_data_dic.pkl', 'rb') as f:

        wheat_data_dic = pickle.load(f)

    for idx in wheat_data_dic:

        for jdx in idx['annotations']:

            jdx['category_id'] = int(0)

    return wheat_data_dic
def create_test_datatset():    

    img_dir = '/kaggle/input/global-wheat-detection/test/'

    dataset_dicts = []

    

    for img_path in glob.glob(img_dir + '*.jpg'):

        record = {}

        file_path = img_path

        image_id = img_path.split('/')[-1].split('.')[0]

        record['file_name'] = file_path

        record['image_id'] = image_id

        dataset_dicts.append(record)

    return dataset_dicts

  
classes = le.classes_
from detectron2.data import DatasetCatalog, MetadataCatalog

DatasetCatalog.register("object_detection_train_comp_2", get_train_data)

MetadataCatalog.get("object_detection_train_comp_2").set(thing_classes=["wheat_head"])

od_dataset = MetadataCatalog.get("object_detection_train_comp_2")
od_dataset
# dataset_dicts = get_train_data()

# for d in random.sample(dataset_dicts, 3):

#     plt.figure(figsize=(20,10)) 

#     img = plt.imread(d["file_name"])

#     visualizer = Visualizer(img[:, :, ::-1], metadata=od_dataset, scale=0.5)

#     vis = visualizer.draw_dataset_dict(d)

#     plt.imshow(vis.get_image()[:, :, ::-1])

#     plt.show()
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg, CfgNode
def custom_mapper(dataset_dict):

    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [T.Resize(1200,1200),

                      T.RandomFlip(prob=0.5),

                      T.RandomContrast(0.8, 3),

                      T.RandomBrightness(0.8, 1.6),

                      ]

    

    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))



    annos = [

        utils.transform_instance_annotations(obj, transforms, image.shape[:2])

        for obj in dataset_dict.pop("annotations")

    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])

    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict
class Trainer(DefaultTrainer):



    @classmethod

    def build_train_loader(cls, cfg: CfgNode):

        return build_detection_train_loader(cfg, mapper=custom_mapper)
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("object_detection_train_comp_2",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 8

cfg.MODEL.WEIGHTS = '../input/detectron2/model_final_280758.pkl' # Let training initialize from model zoo

# cfg.MODEL.WEIGHTS =  model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR

cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = Trainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
%load_ext tensorboard

%tensorboard --logdir cfg.OUTPUT_DIR
DatasetCatalog.register("object_detection_test", create_test_datatset)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

cfg.DATASETS.TEST = ("create_test_datatset", )

predictor = DefaultPredictor(cfg)
test_dataset = create_test_datatset()

img_ids = []

pred_string = []



for test_data in test_dataset:

    image_id = test_data['file_name'].split('/')[-1].split('.')[0]

    img = plt.imread(test_data['file_name'])

    outputs = predictor(img)

#     v = Visualizer(img[:, :, ::-1],

#                    metadata=od_dataset, 

#                    scale=0.3, 

                   

#     )

#     v = v.draw_instance_predictions(outputs["instances"].to("cpu") )

#     plt.figure(figsize=(25, 15))

#     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

#     plt.show()

    preds = []

    for box,score in zip(outputs['instances'].get_fields()['pred_boxes'], outputs['instances'].get_fields()['scores']):

        bbox = []

        for idx in range(4):

            bbox.append(box.data[idx].item())

        preds.append("{} {} {} {} {}".format(score.item(), int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])))

        

    pred_string.append(" ".join(preds))

    img_ids.append(image_id)

            
sub={"image_id":img_ids, "PredictionString":pred_string}

sub=pd.DataFrame(sub)
sub.to_csv('/kaggle/working/submission.csv',index=False)