#! /usr/bin/env python



import sys

import os

import numpy as np

import json

import cv2

import pickle

import tensorflow as tf

import keras

from keras.models import load_model

import matplotlib.pyplot as plt



sys.path.append("../input/kerasyolo3/keras-yolo3-master")

sys.path.append("../input/weightedboxesfusion")





from voc import parse_voc_annotation

from yolo import create_yolov3_model, dummy_loss

from generator import BatchGenerator

from utils.utils import normalize, evaluate, makedirs, get_yolo_boxes

from utils.multi_gpu_model import multi_gpu_model

from ensemble_boxes import *



config = tf.compat.v1.ConfigProto(

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)

    # device_count = {'GPU': 1}

)

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)
max_box_per_image = 116





config = {

    "model" : {

        "min_input_size":       512,

        "max_input_size":       512,

        "anchors":              [0.60,1.18, 0.88,0.58, 1.03,0.98, 1.06,1.56, 1.39,2.48, 1.42,1.19, 1.58,0.71, 1.82,1.66, 2.32,1.13],

        "labels":               ["head"]

    },



    "train": {

        "train_image_folder":   "../data/train/",

        "train_annot_folder":   "../data/VOC_annot.pkl",

        "cache_name":           "",



        "train_times":          1,

        "batch_size":           2,

        "ignore_thresh":        0.5,



        "grid_scales":          [1.0,1.0,1.0],

        "obj_scale":            5,

        "noobj_scale":          1,

        "xywh_scale":           1,

        "class_scale":          1,



        "saved_weights_name":   "../input/wheat-yolo3/wheat_v1.4.h5",

        "debug":                True

    },

}

def create_model(

    nb_class, 

    anchors, 

    max_box_per_image, 

    max_grid, batch_size, 

    warmup_batches, 

    ignore_thresh,

    saved_weights_name,

    grid_scales,

    obj_scale,

    noobj_scale,

    xywh_scale,

    class_scale):

    

    template_model, infer_model = create_yolov3_model(

        nb_class            = nb_class, 

        anchors             = anchors, 

        max_box_per_image   = max_box_per_image, 

        max_grid            = max_grid, 

        batch_size          = batch_size, 

        warmup_batches      = warmup_batches,

        ignore_thresh       = ignore_thresh,

        grid_scales         = grid_scales,

        obj_scale           = obj_scale,

        noobj_scale         = noobj_scale,

        xywh_scale          = xywh_scale,

        class_scale         = class_scale

    )  



    # load the pretrained weight if exists, otherwise load the backend weight only 

    print("\nLoading pretrained weights.\n")

    template_model.load_weights(saved_weights_name)



    return infer_model





model = create_model(

    nb_class            = len(config['model']['labels']), 

    anchors             = config['model']['anchors'], 

    max_box_per_image   = max_box_per_image, 

    max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 

    batch_size          = config['train']['batch_size'], 

    warmup_batches      = 0,

    ignore_thresh       = config['train']['ignore_thresh'],

    saved_weights_name  = config['train']['saved_weights_name'],

    grid_scales         = config['train']['grid_scales'],

    obj_scale           = config['train']['obj_scale'],

    noobj_scale         = config['train']['noobj_scale'],

    xywh_scale          = config['train']['xywh_scale'],

    class_scale         = config['train']['class_scale'],

)
import pandas as pd



im_ids = []

output = []

coords = []



test_files = os.listdir("../input/global-wheat-detection/test/")



for f in test_files:

    im = cv2.imread("../input/global-wheat-detection/test/"+f)

    im_ids.append(f.split(".")[0])

    

    boxes_unfiltered = get_yolo_boxes(model, [im], 512, 512, config['model']['anchors'], 0.2, 1.)[0]

        

    pred_boxes = np.array([[int(box.xmin), int(box.ymin), int(box.xmax - box.xmin), \

                    int(box.ymax - box.ymin)] for box in boxes_unfiltered])/1024



    if len(pred_boxes) > 0:

        pred_boxes[:,2] = pred_boxes[:,2] + pred_boxes[:,0]

        pred_boxes[:,3] = pred_boxes[:,3] + pred_boxes[:,1]

    

        labels_list = [[0 for b in boxes_unfiltered]]

        scores_list = [[box.classes[0] for box in boxes_unfiltered]]

        boxes_list = [pred_boxes]

    

        pred_boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, \

                                                           weights=None, iou_thr=0.3, skip_box_thr=0.95)



        pred_boxes = pred_boxes*1024

        pred_boxes[:,2] = pred_boxes[:,2] - pred_boxes[:,0]

        pred_boxes[:,3] = pred_boxes[:,3] - pred_boxes[:,1]



        preds_sorted_idx = np.argsort(scores)[::-1]

        scores_sorted = np.array(scores[preds_sorted_idx])

        preds_sorted = np.array(pred_boxes[preds_sorted_idx])

    else:

        preds_sorted = np.array([])

    

    all_boxes = []

    

    coords.append(preds_sorted)

    

    for i,box in enumerate(preds_sorted):

        x = [scores[i], int(box[0]), int(box[1]), int(box[2]), int(box[3])]

        x = [str(i) for i in x]

        x = " ".join(x)

        all_boxes.append(x)

    

    output.append(" ".join(all_boxes))

    

def plotboxes(idx):

    im = cv2.imread("../input/global-wheat-detection/test/"+test_files[idx])

    for box in coords[idx]:

        box = [int(x) for x in box]

        cv2.rectangle(im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)



    plt.figure(figsize=(16,16))

    plt.imshow(im)
plotboxes(0)
plotboxes(4)
plotboxes(7)
sub = pd.DataFrame({'image_id':im_ids, 'PredictionString':output})

sub.head()

sub.to_csv("submission.csv", index=False)