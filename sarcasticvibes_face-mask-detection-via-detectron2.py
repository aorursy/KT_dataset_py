# install dependencies: (use cu101 because colab has CUDA 10.1)
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# install detectron2:
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
import torch
torch.cuda.get_device_name()
import os
import pandas as pd
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
import random
import cv2
import random
import matplotlib.pyplot as plt
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
df = pd.read_csv('../input/face-mask-detection-dataset/train.csv')
df.head()
df.describe()
df.classname.unique()
df.groupby('classname').count()
len(df.name.unique())
df.groupby('name').count()
img_folder_dir = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images'
categories = {j:i for i, j in enumerate(df.classname.unique())}
categories
def get_train_dataset():
  train_data = []
  for img in df.name.unique():
    record = {}
    image_id = img[:-4] if '.jpeg' not in img else img[:-5]
    height, width, _ = np.array(Image.open(f'{img_folder_dir}/{img}')).shape
    record['file_name'] = f'{img_folder_dir}/{img}'
    record["image_id"] = image_id
    record["height"] = height
    record["width"] = width

    objs = []
    d = df[df['name'] == img]
    for _, row in d.iterrows():

      xmin = min(row.x1, row.y1)
      ymin = min(row.x2, row.y2)
      xmax = max(row.x1, row.y1)
      ymax = max(row.x2, row.y2)

      poly = [
          (xmin, ymin), (xmax, ymin), 
          (xmax, ymax), (xmin, ymax)
      ]
      poly = [p for x in poly for p in x]

      obj = {
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [poly],
        "category_id": categories[row.classname],
        "iscrowd": 0
      }
      objs.append(obj)
    record["annotations"] = objs
    train_data.append(record)
  return train_data
def get_test_dataset():
  d = pd.read_csv('../input/face-mask-detection-dataset/submission.csv')
  test_data = []
  for img in d.name.unique():
    record = {}
    image_id = img[:-4] if '.jpeg' not in img else img[:-5]
    if ('jpe' in img and 'jpeg' not in img):
      img = (img + 'g') 
    height, width, _ = np.array(Image.open(f'{img_folder_dir}/{img}')).shape
    record['file_name'] = f'{img_folder_dir}/{img}'
    record["image_id"] = image_id
    record["height"] = height
    record["width"] = width
    record["annotations"] = None
    test_data.append(record)
  return test_data
#dataset_dicts = get_train_dataset()
d="train"
DatasetCatalog.register("Face_Mask_Detection_TrainingSet", lambda d=d: get_train_dataset())
MetadataCatalog.get("Face_Mask_Detection_TrainingSet").set(thing_classes=[class_ for class_ in df.classname.unique()])
face_metadata = MetadataCatalog.get("Face_Mask_Detection_TrainingSet")
d="test"
DatasetCatalog.register("Face_Mask_Detection_TestSet", lambda d=d: get_test_dataset())
MetadataCatalog.get("Face_Mask_Detection_TestSet").set(thing_classes=[class_ for class_ in df.classname.unique()])
face_metadata = MetadataCatalog.get("Face_Mask_Detection_TestSet")
# To check if the dataloder function works properly
#for d in random.sample(dataset_dicts, 1):
#    img = cv2.imread(d["file_name"])
#    visualizer = Visualizer(img[:, :, ::-1], metadata=face_metadata, scale = .5)
#    vis = visualizer.draw_dataset_dict(d)
#    img = list(v.get_image()[:, :, ::-1])
#    plt.imshow(img)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Face_Mask_Detection_TrainingSet",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0002  # pick a good LR
cfg.SOLVER.MAX_ITER = 4800
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.OUTPUT_DIR = f'../output/kaggle/working'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
cfg.MODEL.WEIGHTS = f'../output/kaggle/working/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("Face_Mask_Detection_TestSet", )
predictor = DefaultPredictor(cfg)
test_dataset_dicts = get_test_dataset()
# Randomly selecting an image from the test set and drawing the predicted bounding boxes and labels on it
from detectron2.utils.visualizer import ColorMode
for d in random.sample(test_dataset_dicts, 1):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=face_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = list(v.get_image()[:, :, ::-1])
    plt.imshow(img)
categories = {j:i for i,j in categories.items()}
ans = {'name':[], 'x1':[], 'x2':[], 'y1':[],'y2':[],'classname':[]}
for record in tqdm(test_dataset_dicts, total=len(test_dataset_dicts)):
  im = cv2.imread(record["file_name"])
  #cv2_imshow(im)
  outputs = predictor(im)
  outputs = outputs['instances'].to('cpu')
  pred_boxes = outputs.pred_boxes
  pred_labels = outputs.pred_classes
  for box, label in zip(pred_boxes, pred_labels):
    bbox = np.array(box)
    a = record['file_name'][-3:] if 'jpeg' not in record['file_name'] else 'jpeg'
    ans['name'].append(f'{record["image_id"]}.{a}')
    ans['x1'].append(bbox[0])
    ans['x2'].append(bbox[2])
    ans['y1'].append(bbox[1])
    ans['y2'].append(bbox[3])
    #print(label)
    ans['classname'].append(categories[int(label)])
ans = pd.DataFrame(ans)
ans
ans.to_csv(f'../output/kaggle/working/submission.csv')