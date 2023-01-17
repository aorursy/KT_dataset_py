!cp -r ../input/mmdetv2/cocoapi/cocoapi ./
!cp -r ../input/mmdetv2models/mmdetection/mmdetection ./
!cp -r ../input/mmdetv2models/runtime.txt ./mmdetection/requirements
!pip install ../input/mmdetv2/addict-2.2.1-py3-none-any.whl 
!pip install ../input/mmdetv2/terminal-0.4.0-py3-none-any.whl
!pip install ../input/mmdetv2/pytest_runner-5.2-py2.py3-none-any.whl
!pip install ../input/mmdetv2/yapf-0.30.0-py2.py3-none-any.whl
!pip install ../input/mmdetv2models/mmcv-full-1.0.2/mmcv-full-1.0.2
%cd ./cocoapi/pycocotools
!make > /dev/null
!make install > /dev/null
!python setup.py install > /dev/null
import pycocotools
%cd ../..
%cd mmdetection
# ! pip install -r requirements/build.txt
! pip install -v -e . > /dev/null
%cd ../
import sys
# #print(sys.path)
sys.path.append('mmdetection')
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.models import build_detector
import cv2
import pandas as pd
import numpy as np
import os
import re
import json

from PIL import Image
import matplotlib.pyplot as plt
import torch
# !cp -r /kaggle/input/mmdetv2models/wheat_test_cas.py /kaggle/working/mmdetv2models/mmdetection/mmdetection/configs/wheat
# !cp -r /kaggle/input/mmdetv2models/wheat_gfl_no_aug_test.py /kaggle/working/mmdetv2models/mmdetection/mmdetection/configs/wheat
cfg_path = './mmdetection/configs/wheat/wheat_train_cas.py'
cp_path = '/kaggle/input/mmdetv2models/resnest-cas-0.5mosaic-e36-ap540.pth'
img_list = os.listdir('/kaggle/input/global-wheat-detection/test')
model_init = init_detector(cfg_path, cp_path, device='cuda:0')
score_threshold = 0.3
for img in img_list[:10]:
    # test a single image
  img_path = '/kaggle/input/global-wheat-detection/test/'+img
  result = inference_detector(model_init,img_path)
    # show the results
  show_result_pyplot(model_init, img_path, result, score_thr=score_threshold)
  final_scores = np.array(result[0][:,4])
  print(final_scores.shape)
  print(min(final_scores[final_scores>score_threshold]))
!mkdir -p /kaggle/working/data/annotations
!cp -r /kaggle/input/global-wheat-detection/test ./data/test
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


def gen_test_annotation(test_data_path, annotation_path):
    test_anno_list = []
    for img in os.listdir(test_data_path):
        if img.endswith('jpg'):
            img_info = {}
            img_info['filename'] = img
            img_size = Image.open(os.path.join(test_data_path, img)).size
            img_info['width'] = img_size[0]
            img_info['height'] = img_size[1]
            test_anno_list.append(img_info)
    with open(annotation_path, 'w+') as f:
        json.dump(test_anno_list, f)
import sys
sys.path.insert(0, "/kaggle/input/weightedboxesfusion")
from ensemble_boxes import *

def run_wbf(prediction, image_size=1024, iou_thr=0.4, skip_box_thr=0.32, weights=None):
    boxes = [(prediction[:, :4]/(image_size-1)).tolist()]
    scores = [(prediction[:,4]).tolist()]
    labels = [(np.ones(prediction[:,4].shape[0])).tolist() ]
#     boxes, scores, labels = nms(boxes, scores, labels, weights=None, iou_thr=iou_thr)
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def make_predictions(dataset, outputs):
    results = []
    all_boxes = []
    all_scores = []
    image_ids = []
    for images_info, result in zip(dataset.data_infos, outputs):
        boxes, scores, labels = run_wbf(result[0])
        boxes = boxes.astype(np.int32).clip(min=0, max=1023)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[scores >= 0.05].astype(np.int32)
        scores = scores[scores >=float(0.05)]
        if len(boxes)>0:
            result = {
           'image_id': images_info['filename'][:-4],
           'PredictionString': format_prediction_string(boxes, scores)}
        else:
            result = {
           'image_id': images_info['filename'][:-4],
           'PredictionString': ''}
        results.append(result)
        all_boxes.append(boxes)
        all_scores.append(scores)
        image_ids.append(images_info['filename'])
        
        
    return results, image_ids, all_boxes, all_scores
        
DIR_INPUT = './data'
DIR_TEST = f'{DIR_INPUT}/test/'
DIR_ANNO = f'{DIR_INPUT}/annotations'

# DIR_WEIGHTS = '/kaggle/input/mmdetv2models'
# WEIGHTS_FILE = f'{DIR_WEIGHTS}/resnest101-cas-aug-e24-ap47.4.pth'

# test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

# prepare test data annotations
gen_test_annotation(DIR_TEST, DIR_ANNO + '/detection_test.json')
cfg = Config.fromfile(cfg_path)
# cfg.dataset_type = 'WheatDatasetTest
cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 1
cfg.data.test.test_mode = True
# cfg.test_pipeline[0].flip=False
# cfg.data.test.ann_file = DIR_ANNO + '/detection_test.json'
# cfg.data.test.img_prefix = DIR_TEST
cfg.model.pretrained = None
distributed = False
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=distributed,
    shuffle=False)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, cp_path, map_location='cpu')

model.CLASSES = dataset.CLASSES

model = MMDataParallel(model.cuda(0), device_ids=[0])
outputs = single_gpu_test(model, data_loader)
results, image_ids, all_boxes, all_scores = make_predictions(dataset, outputs)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
!rm -rf ./*
# save result
test_df.to_csv('submission.csv', index=False)
test_df.head()
import cv2
import matplotlib.pyplot as plt
def draw_rect(img, bboxes, scores,color=None):
    img = img.copy()
    bboxes = bboxes[:, :4]
    bboxes = bboxes.reshape(-1, 4)
    for bbox,score in zip(bboxes,scores):
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
        img = cv2.rectangle(img.copy(), pt1, pt2, color, int(max(img.shape[:2]) / 200))
        cv2.putText(img, '%.2f'%(score), pt1, cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)                 
    return img

# print(image_ids[0])
# print(all_boxes[0])
fig, ax = plt.subplots(10, 1, figsize=(160, 80))
for i in range(10):
    im0 = cv2.imread('/kaggle/input/global-wheat-detection/test/'+str(image_ids[i]))[:,:,::-1]
    box0 = all_boxes[i]
    box0[:,2] = box0[:,2]+box0[:,0]
    box0[:,3] = box0[:,1]+box0[:,3]
    score0 = np.array(all_scores[i])
    img = draw_rect(im0,np.array(box0),score0, color=(255,0,0))
    ax[i].imshow(img)