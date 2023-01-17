# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# # install dependencies: (use cu101 because colab has CUDA 10.1)

!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 

!pip install cython pyyaml==5.1

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

!gcc --version

# #opencv is pre-installed on colab
pd.read_csv('../input/submission-files/submission (2).csv').to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')
# install detectron2:

!pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
# Importing Necessary Libraries

import tensorflow as tf

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import plotly.express as px

from ast import literal_eval

import plotly.graph_objects as go



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
# # Reading Dataset

dataset = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

dataset
# Making Found columns for bounding box

dataset[['x', 'y', 'w', 'h']] = pd.DataFrame(dataset.bbox.str.strip('[]').str.split(',').tolist())



# Change datatype of columns to float

dataset[['x', 'y', 'w', 'h']] = dataset[['x', 'y', 'w', 'h']].astype(float)



# Adding extenson to the image_id column

dataset['image_id'] = dataset['image_id'] + '.jpg'
def get_bbox_area(bbox):

    bbox = bbox.strip('[]').split(',')

    return float(bbox[2]) * float(bbox[3])

dataset['bbox_area'] = dataset['bbox'].apply(get_bbox_area)
dataset
dataset.describe()
dataset.info()
print(f"There are total {dataset['image_id'].nunique()} number of unique image")
fig = px.histogram(dataset, x="source")

fig.show()
fig = px.histogram(dataset, x="bbox_area")

fig.show()
fig = px.histogram(dataset, x="bbox_area", color="source")

fig.show()
fig = go.Figure(data=[go.Histogram(x=dataset['image_id'].value_counts())])

fig.show()
def show_images(images, num = 5):

    

    images_to_show = np.random.choice(images, num)



    for image_id in images_to_show:



        image_path = os.path.join('/kaggle/input/global-wheat-detection/train/', image_id)

        image = Image.open(image_path)



        # get all bboxes for given image in [xmin, ymin, width, height]

        bboxes = [literal_eval(box) for box in dataset[dataset['image_id'] == image_id]['bbox']]



        # visualize them

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:    

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)



        plt.figure(figsize = (15,15))

        plt.imshow(image)

        plt.show()

        
show_images(dataset['image_id'].unique(), num = 1)
dataset['category_id'] = list(range(0, dataset.shape[0]))

dataset['image_category_id'] = dataset.groupby(['image_id']).ngroup()

dataset
import ast 



dict_dataset = []

def get_dataset_dics(img_dir):

    for i in dataset.groupby('image_id'):

        ann_lst = []

        

        for b in i[1]['bbox']:

            

            b = ast.literal_eval(b)

            

            ann_dict = {'bbox': [b[0], b[1], b[2], b[3]],

           'bbox_mode': BoxMode.XYWH_ABS,

           'category_id': 0,

           'iscrowd': 0}

            

            ann_lst.append(ann_dict)

            

       



    image_dict = {'annotations': ann_lst,

         'file_name': '../input/global-wheat-detection/train/'+pd.DataFrame(i[1]['image_id'].values)[0][0],

         'height': pd.DataFrame(i[1]['height'].values)[0][0],

         'image_id': pd.DataFrame(i[1]['image_category_id'].values)[0][0],

         'width': pd.DataFrame(i[1]['width'].values)[0][0]}

       

    dict_dataset.append(image_dict)

    return dict_dataset



dict_dataset = get_dataset_dics(dataset['image_id'])
from detectron2.data import DatasetCatalog, MetadataCatalog

for d in ["train"]:

   DatasetCatalog.register("wheat_" + d, lambda d=d: get_dataset_dics(dataset['image_id']))

   MetadataCatalog.get("wheat_" + d).set(thing_classes=["wheat_"])

   balloon_metadata = MetadataCatalog.get("wheat_train")

    

dataset_dicts = get_dataset_dics(dataset['image_id'])

for d in dataset_dicts:

    img = cv2.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=1.5)

    vis = visualizer.draw_dataset_dict(d)

    plt.figure(figsize = (15,15))

    plt.imshow(vis.get_image()[:, :, ::-1])

    plt.show()

    break
#from detectron2.engine import DefaultTrainer

#from detectron2.config import get_cfg



#cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

#cfg.DATASETS.TRAIN = ("wheat_train",)

#cfg.DATASETS.TEST = ()

#cfg.DATALOADER.NUM_WORKERS = 2

#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo

#cfg.SOLVER.IMS_PER_BATCH = 2

#cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR

#cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset

#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)

#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)



#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#trainer = DefaultTrainer(cfg) 

#trainer.resume_or_load(resume=False)

#trainer.train()
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("wheat_train",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo



cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR

cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset

cfg.SOLVER.WEIGHT_DECAY = 0.0001







cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

cfg.MODEL.WEIGHTS = os.path.join("../input/model-pretrained/output/model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

cfg.DATASETS.TEST = ("wheat_val", )

predictor = DefaultPredictor(cfg)
final_outputs = []

for i in ['796707dd7', 'cc3532ff6', '51f1be19e', '51b3e36ab', 'f5a1f0358', 'aac893a91', '348a992bb', '2fd875eaa', 'cb8d261a3', '53f253011']:

    image = cv2.imread(f'/kaggle/input/global-wheat-detection/test/{i}.jpg')

    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1],

                   metadata=balloon_metadata, 

                   scale=5.0, 

    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    img = np.array(v.get_image()[:, :, ::-1])

    final_outputs.append(outputs)

    plt.figure(figsize = (15,15))

    plt.imshow(img)

    plt.show()
predictions_col = []

for i in final_outputs:

    final_column_string = ""

    for n in range(0, len(i['instances'].pred_boxes.tensor.tolist())):

        x1 = i['instances'].pred_boxes.tensor.tolist()[n][0]

        x2 = i['instances'].pred_boxes.tensor.tolist()[n][1]

        y1 = i['instances'].pred_boxes.tensor.tolist()[n][2]

        y2 = i['instances'].pred_boxes.tensor.tolist()[n][3]

        score = i['instances'].scores.tolist()[n]

        prediction_string = f"{score} {int(x1)} {int(x2)} {int(y1)} {int(y2)} "

        print(prediction_string)

        final_column_string += prediction_string

    print('\n' + final_column_string)

    predictions_col.append(final_column_string)

    print('---'*30)
submission = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')

submission
submission['PredictionString'] = predictions_col

submission
submission.to_csv('submission.csv', index=False)