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
!pip install '/kaggle/input/torch-15/torch-1.5.0cu101-cp37-cp37m-linux_x86_64.whl'
!pip install '/kaggle/input/torch-15/torchvision-0.6.0cu101-cp37-cp37m-linux_x86_64.whl'
!pip install '/kaggle/input/torch-15/yacs-0.1.7-py3-none-any.whl'
!pip install '/kaggle/input/torch-15/fvcore-0.1.1.post200513-py3-none-any.whl'
!pip install '/kaggle/input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl'
!pip install '/kaggle/input/detectron2/detectron2-0.1.3cu101-cp37-cp37m-linux_x86_64.whl'
# install dependencies: (use cu101 because colab has CUDA 10.1)
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab
# install detectron2:
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import gc
import os
from glob import glob
import cv2

from PIL import Image
import random
from collections import deque, defaultdict
from multiprocessing import Pool, Process
from functools import partial

import pycocotools
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.data import datasets, DatasetCatalog, MetadataCatalog
def display_feature(df, feature):
    
    plt.figure(figsize=(15,8))
    ax = sns.countplot(y=feature, data=df, order=df[feature].value_counts().index)

    for p in ax.patches:
        ax.annotate('{:.2f}%'.format(100*p.get_width()/df.shape[0]), (p.get_x() + p.get_width() + 0.02, p.get_y() + p.get_height()/2))

    plt.title(f'Distribution of {feature}', size=25, color='b')    
    plt.show()
MAIN_PATH = '/kaggle/input/global-wheat-detection'
TRAIN_IMAGE_PATH = os.path.join(MAIN_PATH, 'train/')
TEST_IMAGE_PATH = os.path.join(MAIN_PATH, 'test/')
TRAIN_PATH = os.path.join(MAIN_PATH, 'train.csv')
SUB_PATH = os.path.join(MAIN_PATH, 'sample_submission.csv')

SEED_COLOR = 37
NUMBER_TRAIN_SAMPLE = -1
MODEL_PATH = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
WEIGHT_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
EPOCH = 100

train_img = glob(f'{TRAIN_IMAGE_PATH}/*.jpg')
test_img = glob(f'{TEST_IMAGE_PATH}/*.jpg')

print(f'Number of train image:{len(train_img)}, test image:{len(test_img)}')
sub_df = pd.read_csv(SUB_PATH)
sub_df.tail()
train_df = pd.read_csv(TRAIN_PATH)
train_df.head()
list_source = train_df['source'].unique().tolist()
print(list_source)
display_feature(train_df, 'source')
%%time

image_unique = train_df['image_id'].unique()
image_unique_in_train_path = [i for i in image_unique if i + '.jpg' in os.listdir(TRAIN_IMAGE_PATH)]

print(f'Number of image unique: {len(image_unique)}, in train path: {len(image_unique_in_train_path)}')

del image_unique, image_unique_in_train_path
gc.collect()
def list_color(seed):
    class_unique = sorted(train_df['source'].unique().tolist())
    dict_color = dict()
    random.seed(seed)
    for classid in class_unique:
        dict_color[classid] = random.sample(range(256), 3)
    
    return dict_color


def display_image(df, folder, num_img=3):
    
    if df is train_df:
        dict_color = list_color(SEED_COLOR)
        
    for i in range(num_img):
        fig, ax = plt.subplots(figsize=(15, 15))
        img_random = random.choice(df['image_id'].unique())
        assert (img_random + '.jpg') in os.listdir(folder)
        
        img_df = df[df['image_id']==img_random]
        img_df.reset_index(drop=True, inplace=True)
        
        img = cv2.imread(os.path.join(folder, img_random + '.jpg'))
        for row in range(len(img_df)):
            source = img_df.loc[row, 'source']
            box = img_df.loc[row, 'bbox'][1:-1]
            box = list(map(float, box.split(', ')))
            x, y, w, h = list(map(int, box))
            if df is train_df:
                cv2.rectangle(img, (x, y), (x+w, y+h), dict_color[source], 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
        ax.set_title(f'{img_random} have {len(img_df)} bbox')
        ax.imshow(img)   
        
    plt.show()        
    plt.tight_layout()
    
display_image(train_df, TRAIN_IMAGE_PATH)    
%%time

def wheat_dataset(df, folder, is_train, img_unique):
    img_id, img_name = img_unique
    if is_train:
        img_group = df[df['image_id']==img_name].reset_index(drop=True)
        record = defaultdict()
        img_path = os.path.join(folder, img_name+'.jpg')
        
        record['file_name'] = img_path
        record['image_id'] = img_id
        record['height'] = int(img_group.loc[0, 'height'])
        record['width'] = int(img_group.loc[0, 'width'])
        
        annots = deque()
        for _, ant in img_group.iterrows():
            source = ant.source
            annot = defaultdict()
            box = ant.bbox[1:-1]
            box = list(map(float, box.split(', ')))
            x, y, w, h = list(map(int, box))
            
            annot['bbox'] = (x, y, x+w, y+h)
            annot['bbox_mode'] = BoxMode.XYXY_ABS
            annot['category_id'] = list_source.index(source)
            
            annots.append(dict(annot))
            
        record['annotations'] = list(annots)
    
    else:
        img_group = df[df['image_id']==img_name].reset_index(drop=True)
        record = defaultdict()
        img_path = os.path.join(folder, img_name+'.jpg')
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        record['file_name'] = img_path
        record['image_id'] = img_id
        record['height'] = int(h)
        record['width'] = int(w)
    
    return dict(record)



def wheat_parallel(df, folder, is_train):
    
    if is_train:
        if NUMBER_TRAIN_SAMPLE != -1:
            df = df[:NUMBER_TRAIN_SAMPLE]
        
    pool = Pool()
    img_uniques = list(zip(range(df['image_id'].nunique()), df['image_id'].unique()))
    func = partial(wheat_dataset, df, folder, is_train)
    detaset_dict = pool.map(func, img_uniques)
    pool.close()
    pool.join()
    
    return detaset_dict
for d in ['train', 'test']:
    DatasetCatalog.register(f'wheat_{d}', lambda d=d: wheat_parallel(train_df if d=='train' else sub_df, 
                                                                     TRAIN_IMAGE_PATH if d=='train' else TEST_IMAGE_PATH,
                                                                     True if d=='train' else False))
    MetadataCatalog.get(f'wheat_{d}').set(thing_classes=list_source)
    
micro_metadata = MetadataCatalog.get('wheat_train')
def visual_train(dataset, n_sampler=10):
    for sample in random.sample(dataset, n_sampler):
        img = cv2.imread(sample['file_name'])
        v = Visualizer(img[:, :, ::-1], metadata=micro_metadata, scale=0.5)
        v = v.draw_dataset_dict(sample)
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()
        
train_dataset = wheat_parallel(train_df, TRAIN_IMAGE_PATH, True)        
visual_train(train_dataset)
%%time

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list_source)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

cfg.DATASETS.TRAIN = ('wheat_train',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LS = 0.00025
cfg.SOLVER.MAX_ITER = 5000

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

gc.collect()
import torch
#torch.cuda.empty_cache()
%%time
trainer.train()

gc.collect()

cfg.DATASETS.TEST = ('wheat_test',)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list_source)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predict = DefaultPredictor(cfg)

%%time

def visual_predict(dataset):
    for sample in dataset:
        img = cv2.imread(sample['file_name'])
        output = predict(img)
        
        v = Visualizer(img[:, :, ::-1], metadata=micro_metadata, scale=0.5)
        v = v.draw_instance_predictions(output['instances'].to('cpu'))
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()

test_dataset = wheat_parallel(sub_df, TEST_IMAGE_PATH, False)
visual_predict(test_dataset)

def submit():
    for idx, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
        img_path = os.path.join(TEST_IMAGE_PATH, row.image_id+'.jpg')
        img = cv2.imread(img_path)
        outputs = predict(img)['instances']
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().detach().numpy()
        list_str = []
        for box, score in zip(boxes, scores):
            box[3] -= box[1]
            box[2] -= box[0]
            box = list(map(int, box))
            score = round(score, 4)
            list_str.append(score) 
            list_str.extend(box)
        sub_df.loc[idx, 'PredictionString'] = ' '.join(map(str, list_str))
    
    return sub_df

sub_df = submit()    
sub_df.to_csv('submission.csv', index=False)
sub_df