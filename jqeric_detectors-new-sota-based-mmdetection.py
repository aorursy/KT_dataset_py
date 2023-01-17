# install orderedset
!cp -r ../input/orderedset/ordered-set-4.0.2 ./
%cd ordered-set-4.0.2/
!pip install -e .
%cd ..
!pip install ../input/cu101torch140/torch-1.4.0-cp37-cp37m-linux_x86_64.whl
!pip install ../input/torchvision050cp37/torchvision-0.5.0-cp37-cp37m-linux_x86_64.whl
!pip install ../input/mmcvwhl/addict-2.2.1-py3-none-any.whl
!pip install ../input/mmdetection20-5-13/terminal-0.4.0-py3-none-any.whl
!pip install ../input/mmdetection20-5-13/terminaltables-3.1.0-py3-none-any.whl
!pip install ../input/pytestrunner/pytest_runner-5.2-py2.py3-none-any.whl # new
!pip install ../input/cityscapesscripts150/cityscapesScripts-1.5.0-py3-none-any.whl
!pip install ../input/imagecorruptions/imagecorruptions-1.1.0-py3-none-any.whl
!pip install ../input/asynctest/asynctest-0.13.0-py3-none-any.whl
!pip install ../input/codecov/codecov-2.1.7-py2.py3-none-any.whl
!pip install ../input/ubelt9/ubelt-0.9.1-py3-none-any.whl
!pip install ../input/kwarray/kwarray-0.5.8-py2.py3-none-any.whl
!pip install ../input/xdoctest/xdoctest-0.12.0-py2.py3-none-any.whl
!cp -r ../input/mmcv60 ./
%cd mmcv60/
!pip install -e .
%cd ..
import sys
sys.path.append('mmcv60') # To find local version
!cp -r ../input/detors ./mmdetection
!rm -rf ./mmdetection/mmdet/apis/inference.py
!cp  ../input/mmdetapisinference/inference.py ./mmdetection/mmdet/apis/inference.py
%cd mmdetection
!cp -r ../../input/mmdetection20-5-13/cocoapi/cocoapi .
%cd cocoapi/PythonAPI
!make
!make install
!python setup.py install
%cd ../..
!pip install -v -e .

%cd ../
sys.path.append('mmdetection') # To find local version
# add to sys python path for pycocotools
sys.path.append('/opt/conda/lib/python3.7/site-packages/pycocotools-2.0-py3.7-linux-x86_64.egg') # To find local version
import mmcv

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset

import pandas as pd
import os
import json

from PIL import Image
import matplotlib.pyplot as plt
import torch
!cp  ../input/detectors2/WheatDetectoRS_mstrain_400_1200_r50_40e.py ./mmdetection/configs/DetectoRS/
config = './mmdetection/configs/DetectoRS/WheatDetectoRS_mstrain_400_1200_r50_40e.py'
checkpoint = '/kaggle/input/detectors2/epoch_40.pth'
model = init_detector(config, checkpoint, device='cuda:0')
# model = init_detector(config, checkpoint, device='cpu')
import cv2
img = '/kaggle/input/global-wheat-detection/test/2fd875eaa.jpg'
result = inference_detector(model, img)
# img = cv2.imread(img)
print(type(img))
show_result_pyplot(img, result,['wheat'], score_thr=0.3)
result[0][0].shape
!mkdir  mmdetection/data/
!mkdir  mmdetection/data/Wheatdetection
!mkdir  mmdetection/data/Wheatdetection/annotations
!cp -r ../input/global-wheat-detection/test mmdetection/data/Wheatdetection/test

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
def gen_test_annotation(test_data_path, annotation_path):
    test_anno_dict = {}
    test_anno_dict["info"] = "jianqiu created"
    test_anno_dict["license"] = ["license"]
    id = 0
    test_anno_list = []
    for img in os.listdir(test_data_path):
        if img.endswith('jpg'):
            id += 1
            img_info = {}
            img_size = Image.open(os.path.join(test_data_path, img)).size
            img_info['height'] = img_size[1]
            img_info['width'] = img_size[0]
            img_info['id'] = id            
            img_info['file_name'] = img
            test_anno_list.append(img_info)
    test_anno_dict["images"] = test_anno_list
    test_anno_dict["categories"] = [
    {
      "id": 1,
      "name": "wheat"
    }
  ]
    with open(annotation_path, 'w+') as f:
        json.dump(test_anno_dict, f)
DIR_INPUT = '/kaggle/working/mmdetection/data/Wheatdetection'
DIR_TEST = f'{DIR_INPUT}/test'
DIR_ANNO = f'{DIR_INPUT}/annotations'

DIR_WEIGHTS = '/kaggle/input/detectors2'
WEIGHTS_FILE = f'{DIR_WEIGHTS}/epoch_40.pth'

# prepare test data annotations
gen_test_annotation(DIR_TEST, DIR_ANNO + '/detection_test.json')
config_file = '/kaggle/input/detestorstest/WheatDetectoRS_mstrain_400_1200_r50_40e.py'
cfg = Config.fromfile(config_file)
cfg.data.test.test_mode = True

distributed = False
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    imgs_per_gpu = 1,
    workers_per_gpu=1,
    dist=distributed,
    shuffle=False)
!mkdir -p /root/.cache/torch/checkpoints/
!cp ../input/resnet50/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth
# Wbf
!pip install --no-deps '../input/timm-package/timm-0.1.26-py3-none-any.whl' > /dev/null

import sys
sys.path.insert(0, "../input/weightedboxesfusion")
from ensemble_boxes import *
import numpy as np
def run_wbf(prediction, image_size=1024, iou_thr=0.43, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[:, :4]/(image_size-1)).tolist()]
    scores = [(prediction[:,4]).tolist()]
    labels = [(np.ones(prediction[:,4].shape[0])).tolist() ]

    boxes, scores, labels = nms(boxes, scores, labels, weights=None, iou_thr=iou_thr)
    boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, WEIGHTS_FILE, map_location='cpu') # 'cuda:0'

model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader, False)

results = []

for images_info, result in zip(dataset.img_infos, outputs):
    boxes, scores, labels = run_wbf(result[0][0])
#     boxes = result[0][0][:, :4]
#     scores = result[0][0][:, 4]
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    result = {
        'image_id': images_info['filename'][:-4],
        'PredictionString': format_prediction_string(boxes, scores)
    }

    results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

# save result
test_df.to_csv('submission.csv', index=False)
test_df.head()
!rm -rf mmdetection/
!rm -rf mmcv60/
!rm -rf ordered-set-4.0.2
len(results[4]['PredictionString'].split(' '))//5
