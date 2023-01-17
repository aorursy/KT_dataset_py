import os
!pip install  '/kaggle/input/torch-15/torch-1.5.0cu101-cp37-cp37m-linux_x86_64.whl'

!pip install  '/kaggle/input/torch-15/torchvision-0.6.0cu101-cp37-cp37m-linux_x86_64.whl'

!pip install  '/kaggle/input/torch-15/yacs-0.1.7-py3-none-any.whl'

!pip install  '/kaggle/input/torch-15/fvcore-0.1.1.post200513-py3-none-any.whl'

!pip install  '/kaggle/input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl'

!pip install  '/kaggle/input/detectron2/detectron2-0.1.3cu101-cp37-cp37m-linux_x86_64.whl'
import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()



# import some common libraries

import numpy as np

import pandas as pd

from multiprocessing import Pool, Process

from functools import partial

from collections import deque, defaultdict

import cv2

import glob

# import some common detectron2 utilities

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog

from tqdm.notebook import tqdm
MAIN_PATH = '/kaggle/input/global-wheat-detection'

TRAIN_IMAGE_PATH = os.path.join(MAIN_PATH, 'train/')

TEST_IMAGE_PATH = os.path.join(MAIN_PATH, 'test/')

TRAIN_PATH = os.path.join(MAIN_PATH, 'train.csv')

SUB_PATH = os.path.join(MAIN_PATH, 'sample_submission.csv')

PADDING = 5





MODEL_PATH = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'

WEIGHT_PATH = '/kaggle/input/detectron2-faster-rcnn-101/model_final_971ab9.pkl'



sub_df = pd.read_csv(SUB_PATH)

sub_df.tail()
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

            

            if random.random() >= 0.75:

                random_x = random.randint(0, PADDING)       

                if (x+random_x <= int(img_group.loc[0, 'width'])) and (w >= random_x):

                    x += random_x

                    w -= random_x                

            elif random.random() >= 0.75:

                random_y = random.randint(0, PADDING)

                if (y+random_y <= int(img_group.loc[0, 'height'])) and (h >= random_y):

                    y += random_y

                    h -= random_y

            else:

                if random.random() >= 0.75:

                    random_w = random.randint(0, PADDING)

                    if w >= random_w:

                        w -= random_w

                elif random.random() >= 0.75:

                    random_h = random.randint(0, PADDING)

                    if h >= random_h:

                        h -= random_h

                            

            annot['bbox'] = (x, y, x+w, y+h)

            annot['bbox_mode'] = BoxMode.XYXY_ABS

            annot['category_id'] = 0

            

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



def wheat_parallel(df, folder):

      

    pool = Pool()

    img_uniques = list(zip(range(df['image_id'].nunique()), df['image_id'].unique()))

    func = partial(wheat_dataset, df, folder, False)

    detaset_dict = pool.map(func, img_uniques)

    pool.close()

    pool.join()

    

    return detaset_dict



DatasetCatalog.register('wheat_test', lambda d: wheat_parallel(sub_df,TEST_IMAGE_PATH))

MetadataCatalog.get('wheat_test')

    

# micro_metadata = MetadataCatalog.get('wheat_train')


def cfg_test():

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))

    cfg.MODEL.WEIGHTS = '../input/gwdmodels/Retina_Det_model_final.pth'

    cfg.DATASETS.TEST = ('wheat_test',)

    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.45

    

    return cfg



cfg = cfg_test()

predict = DefaultPredictor(cfg)
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
import matplotlib.pyplot as plt

def visual_predict(dataset):

    for sample in dataset:

        img = cv2.imread(sample['file_name'])

        output = predict(img)

        

        v = Visualizer(img[:, :, ::-1], metadata=micro_metadata, scale=0.5)

        v = v.draw_instance_predictions(output['instances'].to('cpu'))

        plt.figure(figsize = (14, 10))

        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

        plt.show()



test_dataset = wheat_parallel(sub_df, TEST_IMAGE_PATH)

# visual_predict(test_dataset)