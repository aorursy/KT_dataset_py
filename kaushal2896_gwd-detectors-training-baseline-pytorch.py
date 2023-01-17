%%capture

!pip install ../input/detectorsdependencies/packages/ordered_set-4.0.2-py2.py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/torch-1.4.0-cp37-cp37m-linux_x86_64.whl

!pip install ../input/detectorsdependencies/packages/torchvision-0.5.0-cp37-cp37m-linux_x86_64.whl

!pip install ../input/detectorsdependencies/packages/addict-2.2.1-py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/terminal-0.4.0-py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/terminaltables-3.1.0-py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/pytest_runner-5.2-py2.py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/cityscapesScripts-1.5.0-py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/imagecorruptions-1.1.0-py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/asynctest-0.13.0-py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/codecov-2.1.7-py2.py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/ubelt-0.9.1-py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/kwarray-0.5.8-py2.py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/xdoctest-0.12.0-py2.py3-none-any.whl

!pip install ../input/detectorsdependencies/packages/mmcv-0.6.0-cp37-cp37m-linux_x86_64.whl



# Setup DetectoRS and pycocotools

!cp -r ../input/detors ./mmdetection



%cd mmdetection

!cp -r ../../input/mmdetection20-5-13/cocoapi/cocoapi .

%cd cocoapi/PythonAPI

!make

!make install

!python setup.py install

%cd ../..

!pip install -v -e .

%cd ..



import sys

sys.path.append('mmdetection') # To find local version of DetectoRS



# add to sys python path for pycocotools

sys.path.append('/opt/conda/lib/python3.7/site-packages/pycocotools-2.0-py3.7-linux-x86_64.egg') # To find local version
import gc



import mmcv



from mmdet import __version__

from mmdet.apis import init_detector, inference_detector, show_result_pyplot

from mmcv import Config, DictAction

from mmdet.models import build_detector

from mmcv.runner import load_checkpoint, init_dist

from mmcv.parallel import MMDataParallel

from mmdet.apis import single_gpu_test

from mmdet.datasets import build_dataloader, build_dataset

from mmdet.apis import set_random_seed, train_detector

from mmdet.models import build_detector

from mmdet.utils import collect_env, get_root_logger





import argparse

import copy

import os

import os.path as osp

import time



import torch

import shutil

import pandas as pd

import os

import json



from PIL import Image

import matplotlib.pyplot as plt

import torch



import numpy as np

import random



import albumentations as A



SEED = 28

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
# Copy the pre-trained ResNet50 as it's used as backbone of DetectoRS

!mkdir -p /root/.cache/torch/checkpoints/

!cp ../input/resnet50/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth
conv_cfg = dict(type='ConvAWS')



# model settings

model = dict(

    type='RecursiveFeaturePyramid',  # Name of the detector, In case of DetectoRS, it is RFP

    rfp_steps=2,

    rfp_sharing=False,

    stage_with_rfp=(False, True, True, True),

    num_stages=3,

    pretrained='torchvision://resnet50',  # Pre-trained ImageNet ResNet50 as a backbone of DetectoRS

    interleaved=True,

    mask_info_flow=True,

    backbone=dict(  # Configuration of the backbone model

        type='ResNet',  # The type of the backbone, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L288 for more details.

        depth=50,  # The depth of backbone, usually it is 50 or 101 for ResNet and ResNext backbones.

        num_stages=4,  # Number of stages of the backbone.

        out_indices=(0, 1, 2, 3),  # The index of output feature maps produced in each stages

        frozen_stages=1,  # The weights in the first 1 stage are fronzen

        conv_cfg=conv_cfg,  

        sac=dict(type='SAC', use_deform=True),  

        stage_with_sac=(False, True, True, True),

        norm_cfg=dict(type='BN', requires_grad=True),  # The config of normalization layers.

        style='pytorch'),  # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv, 'caffe' means stride 2 layers are in 1x1 convs.

    neck=dict(

        type='FPN',  # The neck of detector is FPN.

        in_channels=[256, 512, 1024, 2048],  # The input channels, this is consistent with the output channels of backbone

        out_channels=256, # The output channels of each level of the pyramid feature map

        num_outs=5),  # The number of output scales

    rpn_head=dict(

        type='RPNHead',  

        in_channels=256,

        feat_channels=256,

        anchor_scales=[8],

        anchor_ratios=[0.5, 1.0, 2.0],

        anchor_strides=[4, 8, 16, 32, 64],

        target_means=[.0, .0, .0, .0],

        target_stds=[1.0, 1.0, 1.0, 1.0],

        loss_cls=dict(

            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),

        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),

    bbox_roi_extractor=dict(

        type='SingleRoIExtractor',

        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),

        out_channels=256,

        featmap_strides=[4, 8, 16, 32]),

    bbox_head=[

        dict(

            type='SharedFCBBoxHead',

            num_fcs=2,

            in_channels=256,

            fc_out_channels=1024,

            roi_feat_size=7,

            num_classes=2,

            target_means=[0., 0., 0., 0.],

            target_stds=[0.1, 0.1, 0.2, 0.2],

            reg_class_agnostic=True,

            loss_cls=dict(

                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),

            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),

        dict(

            type='SharedFCBBoxHead',

            num_fcs=2,

            in_channels=256,

            fc_out_channels=1024,

            roi_feat_size=7,

            num_classes=2,

            target_means=[0., 0., 0., 0.],

            target_stds=[0.05, 0.05, 0.1, 0.1],

            reg_class_agnostic=True,

            loss_cls=dict(

                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),

            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),

        dict(

            type='SharedFCBBoxHead',

            num_fcs=2,

            in_channels=256,

            fc_out_channels=1024,

            roi_feat_size=7,

            num_classes=2,

            target_means=[0., 0., 0., 0.],

            target_stds=[0.033, 0.033, 0.067, 0.067],

            reg_class_agnostic=True,

            loss_cls=dict(

                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),

            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))

    ],

    mask_roi_extractor=dict(

        type='SingleRoIExtractor',

        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),

        out_channels=256,

        featmap_strides=[4, 8, 16, 32]),

    mask_head=[

        dict(

            type='HTCMaskHead',

            with_conv_res=False,

            num_convs=4,

            in_channels=256,

            conv_out_channels=256,

            num_classes=2,

            loss_mask=dict(

                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),

        dict(

            type='HTCMaskHead',

            num_convs=4,

            in_channels=256,

            conv_out_channels=256,

            num_classes=2,

            loss_mask=dict(

                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),

        dict(

            type='HTCMaskHead',

            num_convs=4,

            in_channels=256,

            conv_out_channels=256,

            num_classes=2,

            loss_mask=dict(

                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))

    ])



# model training and testing settings

train_cfg = dict(

    rpn=dict(

        assigner=dict(

            type='MaxIoUAssigner',

            pos_iou_thr=0.7,

            neg_iou_thr=0.3,

            min_pos_iou=0.3,

            ignore_iof_thr=-1),

        sampler=dict(

            type='RandomSampler',

            num=256,

            pos_fraction=0.5,

            neg_pos_ub=-1,

            add_gt_as_proposals=False),

        allowed_border=0,

        pos_weight=-1,

        debug=False),

    rpn_proposal=dict(

        nms_across_levels=False,

        nms_pre=2000,

        nms_post=2000,

        max_num=2000,

        nms_thr=0.7,

        min_bbox_size=0),

    rcnn=[

        dict(

            assigner=dict(

                type='MaxIoUAssigner',

                pos_iou_thr=0.5,

                neg_iou_thr=0.5,

                min_pos_iou=0.5,

                ignore_iof_thr=-1),

            sampler=dict(

                type='RandomSampler',

                num=512,

                pos_fraction=0.25,

                neg_pos_ub=-1,

                add_gt_as_proposals=True),

            mask_size=28,

            pos_weight=-1,

            debug=False),

        dict(

            assigner=dict(

                type='MaxIoUAssigner',

                pos_iou_thr=0.6,

                neg_iou_thr=0.6,

                min_pos_iou=0.6,

                ignore_iof_thr=-1),

            sampler=dict(

                type='RandomSampler',

                num=512,

                pos_fraction=0.25,

                neg_pos_ub=-1,

                add_gt_as_proposals=True),

            mask_size=28,

            pos_weight=-1,

            debug=False),

        dict(

            assigner=dict(

                type='MaxIoUAssigner',

                pos_iou_thr=0.7,

                neg_iou_thr=0.7,

                min_pos_iou=0.7,

                ignore_iof_thr=-1),

            sampler=dict(

                type='RandomSampler',

                num=512,

                pos_fraction=0.25,

                neg_pos_ub=-1,

                add_gt_as_proposals=True),

            mask_size=28,

            pos_weight=-1,

            debug=False)

    ],

    stage_loss_weights=[1, 0.5, 0.25])





test_cfg = dict(

    rpn=dict(

        nms_across_levels=False,

        nms_pre=1000,

        nms_post=1000,

        max_num=1000,

        nms_thr=0.7,

        min_bbox_size=0),

    rcnn=dict(

        score_thr=0.001,

        nms=dict(type='nms', iou_thr=0.5),

        max_per_img=100,

        mask_thr_binary=0.5))





# dataset settings

dataset_type = 'CocoDataset'  # Dataset type, this will be used to define the dataset

# data_root = '../input/gwdannotations/coco/'  # Root path of data

data_root = '/kaggle/input/wheatcoco/coco/'  # Root path of data

img_norm_cfg = dict(  # Image normalization config to normalize the input images

    mean=[123.675, 116.28, 103.53],   # Mean values used to pre-training the pre-trained backbone models

    std=[58.395, 57.12, 57.375],    # Standard variance used to pre-training the pre-trained backbone models

    to_rgb=True)  # The channel orders of image used to pre-training the pre-trained backbone models



train_transforms = [

    dict(type='RandomSizedCrop',

        min_max_height=(800, 800),

        height=1024,

        width=1024,

        p=0.5),

    dict(type='OneOf',

         transforms=[

            dict( type='HueSaturationValue',

                  hue_shift_limit=0.2,

                  sat_shift_limit=0.2,

                  val_shift_limit=0.2, p=0.9),

            dict(type='RandomBrightnessContrast',

                  brightness_limit=0.2,

                  contrast_limit=0.2, p=0.9)

         ], p=0.9),

    dict(type='ToGray', p=0.01),

    dict(type='HorizontalFlip', p=0.5),

    dict(type='VerticalFlip', p=0.5),

    dict(type='Resize', height=512, width=512, p=1.0),

    dict(type='Cutout', num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),

]



val_transforms = [

    dict(type='Resize', height=512, width=512, p=1.0)

]



train_pipeline = [

    dict(type='LoadImageFromFile'),

    dict(

        type='LoadAnnotations', with_bbox=True, with_mask=True),

    dict(

        type='Albu',

        transforms=train_transforms,

        bbox_params=dict(

            type='BboxParams',

            format='pascal_voc',

            label_fields=['gt_labels'],

            min_visibility=0.0,

            filter_lost_elements=True),

        keymap={

            'img': 'image',

            'gt_masks': 'masks',

            'gt_bboxes': 'bboxes'

        },

        update_pad_shape=False,

        skip_img_without_anno=True),

    dict(type='RandomFlip', flip_ratio=0.5),

    dict(type='Normalize', **img_norm_cfg),

    dict(type='DefaultFormatBundle'),

    dict(

        type='Collect',

        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),

]

test_pipeline = [

    dict(type='LoadImageFromFile'),

    dict(

        type='MultiScaleFlipAug',

        img_scale=(512, 512),

        flip=False,

        transforms=[

            dict(type='Resize', keep_ratio=True),

            dict(type='RandomFlip', flip_ratio=0.5),

            dict(type='Normalize', **img_norm_cfg),

            dict(type='Pad', size_divisor=32),

            dict(type='ImageToTensor', keys=['img']),

            dict(type='Collect', keys=['img']),

        ])

]

data = dict(

    imgs_per_gpu=1,  # Batch size

    workers_per_gpu=2,

    train=dict(

        type=dataset_type,

        ann_file=data_root + 'annotations/instances_train2017.json',

        img_prefix=data_root + 'images/train2017/',

        pipeline=train_pipeline),

    val=dict(

        type=dataset_type,

        ann_file=data_root + 'annotations/instances_val2017.json',

        img_prefix=data_root + 'images/val2017/',

        pipeline=test_pipeline),

    test=dict(

        type=dataset_type,

        ann_file=data_root + 'annotations/instances_val2017.json',

        img_prefix=data_root + 'images/val2017/',

        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])



# optimizer

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))



# learning policy

lr_config = dict(

    policy='step',

    warmup='linear',

    warmup_iters=500,

    warmup_ratio=1.0 / 3,

    step=[36, 39])



checkpoint_config = dict(interval=1)

# yapf:disable



log_config = dict(

    interval=50,

    hooks=[

        dict(type='TextLoggerHook'),

    ])

# yapf:enable

# runtime settings



total_epochs = 1

dist_params = dict(backend='nccl')

log_level = 'INFO'

work_dir = 'work_dirs'

load_from = None

resume_from = None

workflow = [('train', 1)]





config_dict=dict(

    model=model,

    train_cfg=train_cfg,

    test_cfg=test_cfg ,

    dataset_type=dataset_type,

    data_root=data_root,

    img_norm_cfg=img_norm_cfg,

    train_pipeline=train_pipeline,

    test_pipeline=test_pipeline,

    data=data ,

    evaluation=evaluation ,

    optimizer=optimizer,

    optimizer_config=optimizer_config,

    lr_config=lr_config,

    total_epochs=total_epochs,

    checkpoint_config=checkpoint_config,

    log_config=log_config,

    dist_params=dist_params,

    log_level=log_level,

    load_from =load_from ,

    resume_from=resume_from,

    workflow=workflow,

    gpus = 1,  # Not sure why?. Solved: Because it's used to get GPU IDs

    work_dir = work_dir

)



config = Config(config_dict) 
cfg = config



# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

torch.backends.cudnn.benchmark = True

cfg.gpu_ids = range(1) if cfg.gpus is None else range(cfg.gpus)



# apply the linear scaling rule (https://arxiv.org/abs/1706.02677)

cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8



distributed = False



# create work_dir

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))



# init the logger before other steps

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())



log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)



# init the meta dict to record some important information such as

# environment info and seed, which will be logged

meta = dict()



# log env info

env_info_dict = collect_env()

env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])

dash_line = '-' * 60 + '\n'

logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

meta['env_info'] = env_info



# log some basic info

logger.info(f'Distributed training: {distributed}')

logger.info(f'Config:\n{cfg.pretty_text}')



# set random seeds

logger.info(f'Set random seed to {SEED}, '

            f'deterministic: {True}')

set_random_seed(SEED, deterministic=True)



cfg.seed = SEED

meta['seed'] = SEED



# Now build the training model

model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)



# Build the training dataset

datasets = [build_dataset(cfg.data.train)]



if len(cfg.workflow) == 2:

    val_dataset = copy.deepcopy(cfg.data.val)

    val_dataset.pipeline = cfg.data.train.pipeline

    datasets.append(build_dataset(val_dataset))

if cfg.checkpoint_config is not None:

    # save mmdet version, config file content and class names in

    # checkpoints as meta data

    cfg.checkpoint_config.meta = dict(

        mmdet_version=__version__,

        config=cfg.pretty_text,

        CLASSES=datasets[0].CLASSES)



# add an attribute for visualization convenience

model.CLASSES = datasets[0].CLASSES

train_detector(

    model,

    datasets,

    cfg,

    distributed=distributed,

    validate=True,

    timestamp=timestamp,

    meta=meta)  
# Finally remove this to prevent the unnecessary output

!rm -rf mmdetection/