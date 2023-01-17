!pip install pyyaml>5.1 pycocotools>=2.0.1 --upgrade
# install dependencies: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab
# install detectron2: (colab has CUDA 10.1 + torch 1.5)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
assert torch.__version__.startswith("1.5")
!pip install detectron2==0.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html --no-cache-dir --upgrade
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import pandas as pd
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

IMG_HEIGHT = 1024
IMG_WIDTH = 1024
path = '/kaggle/input/global-wheat-detection/'
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml"))
# cfg.DATASETS.TRAIN = ("wheat",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 750  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (wheat)
cfg.OUTPUT_DIR = '/kaggle/input/faster-rcnn-750-epoch'

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55   # set a custom testing threshold for this model
cfg.DATASETS.TEST = ("wheat_val",)
predictor = DefaultPredictor(cfg)
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
from detectron2.utils.visualizer import ColorMode

df = pd.DataFrame(columns=['image_id', 'PredictionString'])

for img in os.listdir(f'{path}/test'):
    im = cv2.imread(f'{path}/test/{img}', cv2.IMREAD_COLOR)
    im = cv2.resize(im, (1024, 1024))
    outputs = predictor(im)
    outputs = outputs["instances"].to("cpu")
    dict_ = outputs.get_fields()
    str_ = ''
    
    if len(dict_['pred_boxes']):
        for boxes, score in zip(dict_['pred_boxes'], dict_['scores']):
            score_ = score.numpy().item()
            x1, y1, x2, y2 = boxes.numpy().astype(np.int32).tolist()
            xywh = [x1, x2, x2-x1, y2-y1]
            str_ += ''.join([str(round(score_, 4))+' '] + [str(coor) + ' ' for coor in xywh])
        
    temp_df = pd.DataFrame([[img.split('.')[0], str_.strip()]], columns=['image_id', 'PredictionString'])
    df = df.append(temp_df)

df.to_csv('submission.csv', index=False)
# blank_image = np.zeros((1024,1024,3), np.uint8)
# preds = predictor(blank_image)
# preds = preds["instances"].to("cpu")
# dict_ = preds.get_fields()
# len(dict_['pred_boxes'])