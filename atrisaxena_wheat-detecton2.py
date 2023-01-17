!pip install ../input/wheat-detection/torch-1.5.0cu101-cp37-cp37m-linux_x86_64.whl

!pip install ../input/wheat-detection/torchvision-0.6.0cu101-cp37-cp37m-linux_x86_64.whl
!pip install ../input/wheat-detection/Pillow-7.2.0-cp37-cp37m-manylinux1_x86_64.whl

!pip install ../input/wheat-detection/pycocotools-2.0.1-cp37-cp37m-linux_x86_64.whl

!pip install ../input/wheat-detection/yacs-0.1.7-py3-none-any.whl

!pip install ../input/wheat-detection/fvcore-0.1.1.post20200716-py3-none-any.whl

!pip install ../input/wheat-detection/detectron2-0.2cu101-cp37-cp37m-linux_x86_64.whl
from pathlib import Path

DATA_DIR = Path('/kaggle/input/global-wheat-detection')

TRAIN_PATH = Path(DATA_DIR / 'train')

TEST_PATH = Path(DATA_DIR / 'test')



SUB_PATH = Path(DATA_DIR / 'sample_submission.csv')


import torch, torchvision

import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()



import glob



import os

import ntpath

import numpy as np

import cv2

import random

import itertools

import pandas as pd

from tqdm import tqdm

import urllib

import json

import PIL.Image as Image



from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor, DefaultTrainer

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer, ColorMode

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.structures import BoxMode



import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib import rc



%matplotlib inline

%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)



HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]



sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))



rcParams['figure.figsize'] = 12, 8



RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df = pd.read_csv(DATA_DIR / 'train.csv')



train_df.head()
dataset = []

for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):

    image_name = f"{row['image_id']}.jpg"



    bboxes = row['bbox']

    bboxes = bboxes.replace('[', '')

    bboxes = bboxes.replace(']', '')

    bboxes = bboxes.split(',')



    x_min = float(bboxes[0])

    y_min = float(bboxes[1])

    x_max = float(bboxes[2])

    y_max = float(bboxes[3])



    data = {}



    width = row['width']

    height = row['height']



    data['file_name'] = image_name

    data['width'] = width

    data['height'] = height



    data["x_min"] = x_min

    data["y_min"] = y_min

    data["x_max"] = x_max

    data["y_max"] = y_max



    data['class_name'] = 'wheat'

      

    dataset.append(data)

df = pd.DataFrame(dataset)



df.shape

df.head(2)
def annotate_image(annotations, resize=True, path=str(TRAIN_PATH)):

    file_name = annotations.file_name.to_numpy()[0]

    img = cv2.cvtColor(cv2.imread(f'{path}/{file_name}'), cv2.COLOR_BGR2RGB)

    for i, a in annotations.iterrows():

        cv2.rectangle(img, (int(a.x_min), int(a.y_min)), (int(a.x_max) + int(a.x_min), int(a.y_max) + int(a.y_min)), (0, 255, 0), 2)

    if not resize:

        return img

    return cv2.resize(img, (384, 384), interpolation = cv2.INTER_AREA)
img_id = np.random.randint(len(df.file_name.unique())) 

img_df = df[df.file_name == df.file_name.unique()[img_id]]

# img_df



img = annotate_image(img_df, resize=False)

plt.imshow(img)

plt.axis('off')
# import torch, torchvision

sample_images = [annotate_image(df[df.file_name == f]) for f in df.file_name.unique()[:10]]

sample_images = torch.as_tensor(sample_images)



sample_images = sample_images.permute(0, 3, 1, 2)



plt.figure(figsize=(24, 12))

grid_img = torchvision.utils.make_grid(sample_images, nrow=5)



plt.imshow(grid_img.permute(1, 2, 0))

plt.axis('off')
# TRAINING

unique_files = df.file_name.unique()



train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.98), replace=False))

train_df = df[df.file_name.isin(train_files)]

val_df = df[~df.file_name.isin(train_files)]
train_df.shape
val_df.shape
classes = df.class_name.unique().tolist()

classes
def create_dataset_dicts(df, classes):

  dataset_dicts = []

  for image_id, img_name in enumerate(df.file_name.unique()):

    record = {}

    image_df = df[df.file_name == img_name]

    file_path = f'{TRAIN_PATH}/{img_name}'

    record["file_name"] = file_path

    record["image_id"] = image_id

    record["height"] = int(image_df.iloc[0].height)

    record["width"] = int(image_df.iloc[0].width)

    objs = []

    for _, row in image_df.iterrows():

      xmin = int(row.x_min)

      ymin = int(row.y_min)

      xmax = int(row.x_max)

      ymax = int(row.y_max)



      poly = [

          (xmin, ymin), (xmin+xmax, ymin),

          (xmin+xmax, ymin+ymax), (xmin, ymin+ymax)

      ]

      poly = list(itertools.chain.from_iterable(poly))



      obj = {

        "bbox": [xmin, ymin, xmin+xmax, ymin+ymax],

        "bbox_mode": BoxMode.XYXY_ABS,

        "segmentation": [poly],

        "category_id": classes.index(row.class_name),

        "iscrowd": 0

      }

      objs.append(obj)

    record["annotations"] = objs

    dataset_dicts.append(record)

  return dataset_dicts

# register dataset inot the dataset and metadata catalogues

for d in ["train", "val"]:

  DatasetCatalog.register("wheat_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else val_df, classes))

  MetadataCatalog.get("wheat_" + d).set(thing_classes=classes)

  

statement_metadata = MetadataCatalog.get("wheat_train")
dataset_dicts = create_dataset_dicts(train_df, classes)
nrows = 1

ncols = 3



# Index for iterating over images

pic_index = 0





fig = plt.gcf()



fig.set_size_inches(ncols * 8, nrows * 12)



for i, d in enumerate(random.sample(dataset_dicts, 3)):

    

    sp = plt.subplot(nrows, ncols, i + 1, facecolor='red')

    sp.axis('Off')

    

    img = cv2.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=statement_metadata, scale=0.5)

    vis = visualizer.draw_dataset_dict(d)

    plt.imshow(vis.get_image()[:, :, ::-1], interpolation = 'bicubic')
class CocoTrainer(DefaultTrainer):

  @classmethod

  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:

        os.makedirs("coco_eval", exist_ok=True)

        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("wheat_train",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  

cfg.MODEL.WEIGHTS = os.path.join('../input/wheat-detection-model/output/model_final.pth')

cfg.SOLVER.IMS_PER_BATCH = 8

cfg.SOLVER.BASE_LR = 0.0001  

cfg.SOLVER.MAX_ITER = 5000   

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer.train()
print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

cfg.DATASETS.TEST = ("wheat_val", )

evaluator = COCOEvaluator("wheat_val", cfg, False, output_dir="./output/")

val_loader = build_detection_test_loader(cfg, "wheat_val")

inference_on_dataset(trainer.model, val_loader, evaluator)
# Finding wheats in images



cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

predictor = DefaultPredictor(cfg)
test_image_paths = os.listdir(TEST_PATH)



test_image_paths
annotated_results = []



for wheat_image in test_image_paths:

  file_path = f'{TEST_PATH}/{wheat_image}'

  im = cv2.imread(file_path)

  outputs = predictor(im)

  v = Visualizer(

    im[:, :, ::-1],

    metadata=statement_metadata,

    scale=1.,

    instance_mode=ColorMode.IMAGE

  )

  instances = outputs["instances"].to("cpu")

  # instances.remove('pred_masks')

  v = v.draw_instance_predictions(instances)

  result = v.get_image()[:, :, ::-1]

  file_name = ntpath.basename(wheat_image)

  annotated_results.append(result)

img =cv2.cvtColor(annotated_results[0], cv2.COLOR_BGR2RGB)

plt.imshow(img)

plt.axis('off')
img =cv2.cvtColor(annotated_results[1], cv2.COLOR_BGR2RGB)

plt.imshow(img)

plt.axis('off')
sub_df = pd.read_csv(SUB_PATH)

sub_df
def submit():

    for idx, row in tqdm(sub_df.iterrows(), total=len(sub_df)):

        img_path = os.path.join(TEST_PATH, row.image_id + '.jpg')

        

        img = cv2.imread(img_path)

        outputs = predictor(img)['instances']

        #print(outputs)

        # instances.remove('pred_masks')

        #outputs.remove('pred_masks')

        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]

        scores = outputs.scores.cpu().detach().numpy()

        list_str = []

        for box, score in zip(boxes, scores):

            box[3] -= box[1]

            box[2] -= box[0]

            box = list(map(int,box))

            score = round(score, 4)

            list_str.append(score)

            list_str.extend(box)

        sub_df.loc[idx, 'PredictionString'] = ' '.join(map(str, list_str))

        

    return sub_df
subm_df = submit()

subm_df.to_csv('submission.csv', index=False)
subm_df
subm_df['PredictionString'][2]