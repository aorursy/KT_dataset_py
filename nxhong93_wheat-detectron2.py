!pip install '/kaggle/input/torch-15/torch-1.5.0cu101-cp37-cp37m-linux_x86_64.whl'

!pip install '/kaggle/input/torch-15/torchvision-0.6.0cu101-cp37-cp37m-linux_x86_64.whl'

!pip install '/kaggle/input/torch-15/yacs-0.1.7-py3-none-any.whl'

!pip install '/kaggle/input/torch-15/fvcore-0.1.1.post200513-py3-none-any.whl'

!pip install '/kaggle/input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl'

!pip install '/kaggle/input/detectron2/detectron2-0.1.3cu101-cp37-cp37m-linux_x86_64.whl'
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

import gc

import os

import copy

from glob import glob

import cv2

from PIL import Image

import random

from collections import deque, defaultdict

from multiprocessing import Pool, Process

from functools import partial



import torch



import pycocotools

import detectron2

from detectron2.config import get_cfg

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor, DefaultTrainer

from detectron2.utils.visualizer import Visualizer, ColorMode

from detectron2.structures import BoxMode

from detectron2.data import datasets, DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader

from detectron2.data import transforms as T

from detectron2.data import detection_utils as utils

from detectron2.evaluation import COCOEvaluator, verify_results

from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.transforms import TransformGen

from detectron2.utils.logger import setup_logger

setup_logger()



from fvcore.transforms.transform import TransformList, Transform, NoOpTransform

from contextlib import contextmanager



import torch.nn as nn
MAIN_PATH = '/kaggle/input/global-wheat-detection'

TRAIN_IMAGE_PATH = os.path.join(MAIN_PATH, 'train/')

TEST_IMAGE_PATH = os.path.join(MAIN_PATH, 'test/')

TRAIN_PATH = os.path.join(MAIN_PATH, 'train.csv')

SUB_PATH = os.path.join(MAIN_PATH, 'sample_submission.csv')

PADDING = 5



MODEL_USE = 'retinanet'

NUMBER_TRAIN_SAMPLE = -1

if MODEL_USE == 'faster_rcnn':

    MODEL_PATH = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'

    WEIGHT_PATH = '/kaggle/input/detectron2-faster-rcnn-101/model_final_f6e8b1.pkl'

elif MODEL_USE == 'retinanet':

    MODEL_PATH = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'

    WEIGHT_PATH = '/kaggle/input/detectron2-faster-rcnn-101/model_final_971ab9.pkl'
def display_feature(df, feature):

    

    plt.figure(figsize=(15,8))

    ax = sns.countplot(y=feature, data=df, order=df[feature].value_counts().index)



    for p in ax.patches:

        ax.annotate('{:.2f}%'.format(100*p.get_width()/df.shape[0]), (p.get_x() + p.get_width() + 0.02, p.get_y() + p.get_height()/2))



    plt.title(f'Distribution of {feature}', size=25, color='b')    

    plt.show()

    

    

    

def rand_bbox(img, box_size=50):

    

    h, w = img.shape[:2]

    num_rand = np.random.randint(10, 20)

    for num_cut in range(num_rand):

        x_rand, y_rand = random.randint(0, w-box_size), random.randint(0, h-box_size)

        img[x_rand:x_rand+box_size, y_rand:y_rand+box_size, :] = 0

    

    return img
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
num_box = train_df.groupby('image_id')['bbox'].count().reset_index().add_prefix('Number_').sort_values('Number_bbox', ascending=False)

num_box.head()
%%time



image_unique = train_df['image_id'].unique()

image_unique_in_train_path = [i for i in image_unique if i + '.jpg' in os.listdir(TRAIN_IMAGE_PATH)]



print(f'Number of image unique: {len(image_unique)}, in train path: {len(image_unique_in_train_path)}')



del image_unique, image_unique_in_train_path

gc.collect()
def list_color():

    class_unique = sorted(train_df['source'].unique().tolist())

    dict_color = dict()

    for classid in class_unique:

        dict_color[classid] = random.sample(range(256), 3)

    

    return dict_color





def display_image(df, folder, num_img=1, cutmix_prob=0.5):

    

    if df is train_df:

        dict_color = list_color()

        

    for i in range(num_img):

        fig, ax = plt.subplots(figsize=(15, 15))

        img_random = random.choice(df['image_id'].unique())

        assert (img_random + '.jpg') in os.listdir(folder)

        

        img_df = df[df['image_id']==img_random]

        img_df.reset_index(drop=True, inplace=True)

        

        img = cv2.imread(os.path.join(folder, img_random + '.jpg'))

        if random.random() > cutmix_prob:

            img = rand_bbox(img, box_size=50)

                

            

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
class CutMix(Transform):

    

    def __init__(self, box_size=50, prob_cutmix=0.5):

        super().__init__()

        

        self.box_size = box_size

        self.prob_cutmix = prob_cutmix

        

    def apply_image(self, img):

        

        if random.random() > self.prob_cutmix:

            

            h, w = img.shape[:2]

            num_rand = np.random.randint(10, 20)

            for num_cut in range(num_rand):

                x_rand, y_rand = random.randint(0, w-self.box_size), random.randint(0, h-self.box_size)

                img[x_rand:x_rand+self.box_size, y_rand:y_rand+self.box_size, :] = 0

        

        return np.asarray(img)

    

    def apply_coords(self, coords):

        return coords.astype(np.float32)
class DatasetMapper:

    """

    A callable which takes a dataset dict in Detectron2 Dataset format,

    and map it into a format used by the model.



    This is the default callable to be used to map your dataset dict into training data.

    You may need to follow it to implement your own one for customized logic,

    such as a different way to read or transform images.

    See :doc:`/tutorials/data_loading` for details.



    The callable currently does the following:



    1. Read the image from "file_name"

    2. Applies cropping/geometric transforms to the image and annotations

    3. Prepare data and annotations to Tensor and :class:`Instances`

    """



    def __init__(self, cfg, is_train=True):

        if cfg.INPUT.CROP.ENABLED and is_train:

            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)

            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))

        else:

            self.crop_gen = None



#         self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        self.tfm_gens = [T.RandomBrightness(0.1, 1.6),

                         T.RandomContrast(0.1, 3),

                         T.RandomSaturation(0.1, 2),

                         T.RandomRotation(angle=[90, 90]),

                         T.RandomFlip(prob=0.4, horizontal=False, vertical=True),

                         T.RandomCrop('relative_range', (0.4, 0.6)),

                         CutMix()

                        ]



        # fmt: off

        self.img_format     = cfg.INPUT.FORMAT

        self.mask_on        = cfg.MODEL.MASK_ON

        self.mask_format    = cfg.INPUT.MASK_FORMAT

        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON

        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        # fmt: on

        if self.keypoint_on and is_train:

            # Flip only makes sense in training

            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        else:

            self.keypoint_hflip_indices = None



        if self.load_proposals:

            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE

            self.proposal_topk = (

                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN

                if is_train

                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST

            )

        self.is_train = is_train



    def __call__(self, dataset_dict):

        """

        Args:

            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.



        Returns:

            dict: a format that builtin models in detectron2 accept

        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # USER: Write your own image loading if it's not from a file

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        utils.check_image_size(dataset_dict, image)



        if "annotations" not in dataset_dict:

            image, transforms = T.apply_transform_gens(

                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image

            )

        else:

            # Crop around an instance if there are instances in the image.

            # USER: Remove if you don't use cropping

            if self.crop_gen:

                crop_tfm = utils.gen_crop_transform_with_instance(

                    self.crop_gen.get_crop_size(image.shape[:2]),

                    image.shape[:2],

                    np.random.choice(dataset_dict["annotations"]),

                )

                image = crop_tfm.apply_image(image)

            image, transforms = T.apply_transform_gens(self.tfm_gens, image)

            if self.crop_gen:

                transforms = crop_tfm + transforms



        image_shape = image.shape[:2]  # h, w



        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,

        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.

        # Therefore it's important to use torch.Tensor.

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))



        # USER: Remove if you don't use pre-computed proposals.

        if self.load_proposals:

            utils.transform_proposals(

                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk

            )



        if not self.is_train:

            # USER: Modify this if you want to keep them for some reason.

            dataset_dict.pop("annotations", None)

            dataset_dict.pop("sem_seg_file_name", None)

            return dataset_dict



        if "annotations" in dataset_dict:

            # USER: Modify this if you want to keep them for some reason.

            for anno in dataset_dict["annotations"]:

                if not self.mask_on:

                    anno.pop("segmentation", None)

                if not self.keypoint_on:

                    anno.pop("keypoints", None)



            # USER: Implement additional transformations if you have other types of data

            annos = [

                utils.transform_instance_annotations(

                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices

                )

                for obj in dataset_dict.pop("annotations")

                if obj.get("iscrowd", 0) == 0

            ]

            instances = utils.annotations_to_instances(

                annos, image_shape, mask_format=self.mask_format

            )

            # Create a tight bounding box from masks, useful when image is cropped

            if self.crop_gen and instances.has("gt_masks"):

                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = utils.filter_empty_instances(instances)



        # USER: Remove if you don't do semantic/panoptic segmentation.

        if "sem_seg_file_name" in dataset_dict:

            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:

                sem_seg_gt = Image.open(f)

                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")

            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)

            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

            dataset_dict["sem_seg"] = sem_seg_gt

        return dataset_dict
for d in ['train', 'test']:

    DatasetCatalog.register(f'wheat_{d}', lambda d=d: wheat_parallel(train_df if d=='train' else sub_df, 

                                                                     TRAIN_IMAGE_PATH if d=='train' else TEST_IMAGE_PATH,

                                                                     True if d=='train' else False))

    MetadataCatalog.get(f'wheat_{d}')

    

micro_metadata = MetadataCatalog.get('wheat_train')
def visual_train(dataset, n_sampler=1):

    for sample in random.sample(dataset, n_sampler):

        img = cv2.imread(sample['file_name'])

        v = Visualizer(img[:, :, ::-1], metadata=micro_metadata, scale=0.5)

        v = v.draw_dataset_dict(sample)

        plt.figure(figsize = (14, 10))

        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

        plt.show()

        

train_dataset = wheat_parallel(train_df, TRAIN_IMAGE_PATH, True)        

visual_train(train_dataset)
def cfg_setup():

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))

    cfg.MODEL.WEIGHTS = WEIGHT_PATH

    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256



    cfg.DATASETS.TRAIN = ('wheat_train',)

    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 4



    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'

    cfg.SOLVER.BASE_LS = 0.0002

#     cfg.SOLVER.WARMUP_ITERS = 4500

#     cfg.SOLVER.WARMUP_METHOD = "linear"

    cfg.SOLVER.MAX_ITER = 10000

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    

    return cfg







class WheatTrainer(DefaultTrainer):

    

    @classmethod

    def build_train_loader(cls, cfg):

        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg))
%%time



cfg = cfg_setup()

trainer = WheatTrainer(cfg)

trainer.resume_or_load(resume=False)



gc.collect()
%%time

trainer.train()



gc.collect()
def cfg_test():

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

    cfg.DATASETS.TEST = ('wheat_test',)

    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.45

    

    return cfg



cfg = cfg_test()

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