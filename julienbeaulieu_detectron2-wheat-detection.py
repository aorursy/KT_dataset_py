!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
import os
import copy
import cv2

import torch
import pycocotools
import detectron2

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:20]:
        print(os.path.join(dirname, filename))
data_dir = Path('/kaggle/input/global-wheat-detection')
train_img_dir = Path(data_dir / 'train')
test_img_dir = Path(data_dir / 'test')

sub_path = Path(data_dir / 'sample_submission.csv')
df = pd.read_csv(data_dir / 'train.csv')
sub_df = pd.read_csv(sub_path)
df.head()
# Number of instances
df.shape[0]
order = df.source.value_counts().index

plt.figure(figsize=(15, 5))
ax = sns.countplot(data=df, y='source', order=order)

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_width()/df.shape[0]), (p.get_x() + p.get_width() + 0.02, 
                                                                 p.get_y() + p.get_height()/2))
plt.title('Distribution of souce', size=25, color='b')
plt.show()
# Groupby image_id to get an image per row
unique_images = df.groupby('image_id')[['bbox', 'source']].agg(lambda x: list(x)).reset_index()
unique_images.head()
# Number of images
print(f'Number of unique images: {unique_images.shape[0]}')
cnt = {}

for idx, row in unique_images.iterrows(): 
    length = len(unique_images.bbox.iloc[idx])
    image_id = unique_images.image_id.iloc[idx]
    cnt.update({image_id: length})

df_count_bbox = pd.Series(data=cnt)
print(f'Average number of bboxes per image: {df_count_bbox.values.mean()}')
print(f'Max number of bboxes: {df_count_bbox.values.max()}')
print(f'Min number of bboxes: {df_count_bbox.values.min()}')
# Distribution of the number of instances per image
plt.figure(figsize=(10, 5))
sns.distplot(df_count_bbox)
def get_wheat_dict(df, folder):
    
    grps = df['image_id'].unique().tolist()
    df_group = df.groupby('image_id')
    dataset_dicts = []
    
    print('Creating wheat dataset dict...')
    for idx, image_name in enumerate(tqdm(grps)):
        
        # Get all instances of an image
        group = df_group.get_group(image_name)
        
        record = defaultdict()
        
        # Full image path 
        img_path = os.path.join(folder, image_name + '.jpg')
        
        record['height'] = int(group['height'].values[0])
        record['width'] = int(group['width'].values[0])
        record['file_name'] = img_path
        record['image_id'] = idx
        
        objs = []
        
        # Iterate over the group's rows as namedtuples
        for row in group.itertuples():
            
            # Each bbox row is a list of strings - We need it to be a list of ints
            # First we get the string part from the list
            box = row.bbox[1:-1]
            
            # Convert the string to a list of floats
            box = list(map(float, box.split(', ')))
            
            # Convert to int
            x, y, w, h = list(map(int, box))
            
            obj = {
                'bbox': (x, y, x+w, y+h), # change to XYXY format. Original was in XYWH
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': 0, # only 1 category for this dataset
                'iscrowd': 0
                  }
            
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts
# https://www.kaggle.com/nxhong93/wheat-detectron2/

class CutOut(Transform):
    
    def __init__(self, box_size=50, prob_cutmix=0.8):
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

    This is a custom version of the DatasetMapper. The only different with Detectron2's 
    DatasetMapper is that we extract attributes from our dataset_dict. 
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None
        
        self.tfm_gens = [T.RandomBrightness(0.1, 1.6),
                         T.RandomContrast(0.1, 1),
                         T.RandomSaturation(0.1, 1),
                         T.RandomRotation(angle=[90, 90]),
                         T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
                         T.RandomCrop('relative_range', (0.4, 0.6)),
                         CutOut()
                        ]

        # self.tfm_gens = utils.build_transform_gen(cfg, is_train)

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
class WheatTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg))
def train_test_datasets(df, n_sample_size=-1, train_test_split=0.8):
    
    # Option to set a smaller df size for testing
    if n_sample_size == -1:
        n_sample_size = df.shape[0]
    elif n_sample_size != -1:
        n_sample_size = df[:n_sample_size].shape[0]

    # Split df into train / test dataframes
    n_train = round(n_sample_size * train_test_split)
    n_test = n_sample_size - n_train

    df_train = df[:n_train].copy()
    df_test = df[-n_test:].copy()
    
    return df_train, df_test
#df_train, df_test = train_test_datasets(df)
def register_dataset(df, dataset_label='wheat_train', image_dir=train_img_dir):
    
    # Register dataset - if dataset is already registered, give it a new name    
    try:
        DatasetCatalog.register(dataset_label, lambda d=df: get_wheat_dict(df, image_dir))
        MetadataCatalog.get(dataset_label).thing_classes = ['wheat']
    except:
        # Add random int to dataset name to not run into 'Already registered' error
        n = random.randint(1, 1000)
        dataset_label = dataset_label + str(n)
        DatasetCatalog.register(dataset_label, lambda d=df: get_wheat_dict(df, image_dir))
        MetadataCatalog.get(dataset_label).thing_classes = ['wheat']
        
    return MetadataCatalog.get(dataset_label), dataset_label
# Register train dataset
metadata, train_dataset = register_dataset(df)
# Register test dataset
metadata, test_dataset = register_dataset(sub_df, dataset_label='wheat_test', image_dir=test_img_dir)
wheat_dict = get_wheat_dict(df_train, image_dir)
# Visualize image and bbox
import random
for d in random.sample(wheat_dict, 2):
    plt.figure(figsize=(10,10))
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])
MODEL_USE = 'retinanet'
if MODEL_USE == 'faster_rcnn':
    MODEL_PATH = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    WEIGHT_PATH = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
elif MODEL_USE == 'retinanet':
    MODEL_PATH = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
    WEIGHT_PATH = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'

def cfg_setup():
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(WEIGHT_PATH)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'
    cfg.SOLVER.BASE_LS = 0.0002
#     cfg.SOLVER.WARMUP_ITERS = 4500
#     cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.MAX_ITER = 100
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
    return cfg
cfg = cfg_setup()
trainer = WheatTrainer(cfg)    
from detectron2.data import detection_utils as utils

train_data_loader = trainer.build_train_loader(cfg)
data_iter = iter(train_data_loader)
batch = next(data_iter)
rows, cols = 2, 2
plt.figure(figsize=(20,20))

for i, per_image in enumerate(batch[:4]):
    
    plt.subplot(rows, cols, i+1)
    
    # Pytorch tensor is in (C, H, W) format
    img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
    img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

    visualizer = Visualizer(img, metadata=metadata, scale=0.5)

    target_fields = per_image["instances"].get_fields()
    labels = None
    vis = visualizer.overlay_instances(
        labels=labels,
        boxes=target_fields.get("gt_boxes", None),
        masks=target_fields.get("gt_masks", None),
        keypoints=target_fields.get("gt_keypoints", None),
    )
    plt.imshow(vis.get_image()[:, :, ::-1])
trainer.resume_or_load(resume=False)
trainer.train()
def cfg_test():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    
    return cfg

cfg = cfg_test()
predict = DefaultPredictor(cfg)
def visual_predict(dataset):
    for sample in dataset:
        im = cv2.imread(sample['file_name'])
        output = predict(im)
        
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5)
        v = v.draw_instance_predictions(output['instances'].to('cpu'))
        plt.figure(figsize = (10, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()
        
wheat_test_dict = get_wheat_dict(df_test[:50], image_dir)
visual_predict(wheat_test_dict)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    
evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir="./output")
val_loader = build_detection_test_loader(cfg, test_dataset) 
inference_on_dataset(trainer.model, val_loader, evaluator)
def submit():
    for idx, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
        img_path = os.path.join(test_img_dir, row.image_id + '.jpg')
        
        img = cv2.imread(img_path)
        outputs = predict(img)['instances']
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
sub_df = submit()
sub_df.to_csv('submission.csv', index=False)
sub_df