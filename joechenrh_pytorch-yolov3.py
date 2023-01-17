import numpy as np

import cv2

import torch

import csv

import os

import sys



sys.path.insert(0, "/kaggle/input/yolov3")
from models import *

from utils.datasets import *

from utils.utils import *



import csv



import torch

import torch.backends.cudnn as cudnn

import torch.nn as nn

import torch.nn.functional as F



inference_size = 512



device = torch.device('cuda:0')



model = Darknet('/kaggle/input/yolov3/yolov3-spp-customanchor.cfg', inference_size)

model.load_state_dict(torch.load('/kaggle/input/yolov3/best.pt', map_location=device)['model'])

model.to(device).eval()



img_paths = os.listdir("/kaggle/input/global-wheat-detection/test/")

img_paths = [os.path.join("/kaggle/input/global-wheat-detection/test/", img_path) for img_path in img_paths]



img = torch.zeros((1, 3, inference_size, inference_size), device=device)  # init img

_ = model(img.float())



f = open('/kaggle/working/submission.csv','w',encoding='utf-8', newline='')

csv_writer = csv.writer(f)

csv_writer.writerow(["image_id", "PredictionString"])



dataset = LoadImages("/kaggle/input/global-wheat-detection/test/", img_size=inference_size)

for path, img, im0s, _ in dataset:

    img = torch.from_numpy(img).to(device)

    img = img.float() / 255.0

    img = img.unsqueeze(0)



    pred = model(img, False)[0]

    pred = non_max_suppression(pred, 0.3, 0.6, multi_label=False, classes=None, agnostic=False)



    li = []

    for i, det in enumerate(pred):  # detections for image i

        if det is not None and len(det):

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in det:

                li += [float(conf), int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])]

            

    string = ' '.join([str(v) for v in li])

    print(path)

    csv_writer.writerow([os.path.basename(path)[0:-4], string])  

'''

for img_path in img_paths:

    img0 = cv2.imread(img_path)  # BGR

    img = letterbox(img0, new_shape=inference_size)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

    img = np.ascontiguousarray(img)

    

    img = torch.from_numpy(img).to(device)

    img = img.float()

    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:

        img = img.unsqueeze(0)

        

    pred = model(img, False)[0]

    pred = non_max_suppression(pred, 0.3, 0.6, multi_label=False, classes=None, agnostic=False)

    

    li = []

    for i, det in enumerate(pred):  # detections for image i

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls in det:

            li += [float(conf), int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])]

            

    string = ' '.join([str(v) for v in li])

    csv_writer.writerow([os.path.basename(img_path)[0:-4], string])

'''



f.close()