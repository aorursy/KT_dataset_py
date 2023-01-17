import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
train_df = pd.read_csv('../input/global-wheat-detection/train.csv')
train_df.head()
total_data = []

for g in tqdm_notebook(train_df.groupby('image_id')):

    data = {}

    data['filename'] = g[0]

    data['bbox'] = g[1]['bbox'].values

    total_data.append(data)
!pip install '../input/detectron2-wheat/pycocotools-2.0.0-cp37-cp37m-linux_x86_64.whl'
!pip install '../input/detectron2-wheat/yacs-0.1.7-py3-none-any.whl'

!pip install '../input/detectron2-wheat/fvcore-0.1.1.post200513-py3-none-any.whl'

!pip install '../input/detectron2-wheat/detectron2-0.1.2cu101-cp37-cp37m-linux_x86_64.whl'
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



import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np

import cv2

import random

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog
wheat_metadata = MetadataCatalog.get("wheat_train")
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg



cfg = get_cfg()

cfg.merge_from_file('../input/detectron2-wheat/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')   

cfg.DATASETS.TRAIN = ("wheat_train",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 8

cfg.SOLVER.BASE_LR = 0.00025  

cfg.SOLVER.MAX_ITER = 1500   

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

cfg.MODEL.WEIGHTS = '../input/simple-detectron2-training/output/model_final.pth'

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

cfg.DATASETS.TEST = ("wheat_val", )

predictor = DefaultPredictor(cfg)
import glob 

import cv2



image_id = []

PredictionString = []



for f in tqdm_notebook(glob.glob('../input/global-wheat-detection/test/*.jpg')):

    img = cv2.imread(f)

    outputs = predictor(img)

    res = outputs['instances'].to('cpu')

    

    s_pred = ''



    for i, s in enumerate(res.scores.tolist()):

        s_pred += '{:.4f} {} {} {} {} '.format(s,int(res.pred_boxes.tensor.tolist()[i][0]),int(res.pred_boxes.tensor.tolist()[i][1]),

                                                               int(res.pred_boxes.tensor.tolist()[i][2] - res.pred_boxes.tensor.tolist()[i][0]),int(res.pred_boxes.tensor.tolist()[i][3] - res.pred_boxes.tensor.tolist()[i][1]))

    

    

    image_id.append(f.split('/')[-1].split('.')[0])

    PredictionString.append(s_pred)
res_df = pd.DataFrame({'image_id':image_id,'PredictionString':PredictionString})
res_df.head()
res_df.to_csv('submission.csv',index=False)