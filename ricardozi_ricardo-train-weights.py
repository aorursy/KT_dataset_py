!cp -r ../input/yolov5-for-wheat/yolov5_for_test/* /kaggle/working/

!rm -rf ./submission.csv
import torch

import torchvision

from tqdm import tqdm

import pandas as pd

import numpy as np

import os

from glob import glob

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sns

import shutil

sns.set()

import cv2



import argparse



from utils.datasets import *

from utils.utils import *
def detect(save_img=False):

    weights, imgsz = opt.weights,opt.img_size

    source = '/kaggle/input/global-wheat-detection/test'

   

    

    # Initialize

    device = torch_utils.select_device(opt.device)

    half = device.type != 'cpu'

    



    #half = False

    # Load model



    model = torch.load(weights, map_location=device)['model'].to(device).eval()

    if half:

      model.half() 



    dataset = LoadImages(source, img_size=1024)



    t0 = time.time()

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    

    results=[]

    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:

        print(im0s.shape)

        im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)

        image_id = path.split("/")[-1].split(".")[0]

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:

            img = img.unsqueeze(0)

            

        # Inference

        t1 = torch_utils.time_synchronized()

        if True:

            pred = model(img, augment=opt.augment)[0]

            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, merge=opt.merge)

            t2 = torch_utils.time_synchronized()

            box_scores = []

            # Process detections

            for i, det in enumerate(pred):  # detections per image

                p, s, im0 = path, '', im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

                if det is not None and len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():

                        n = (det[:, -1] == c).sum()  # detections per class



                    for *xyxy, conf, cls in det:

                        if True:  # Write to file

                            # xywh = torch.tensor(xyxy).view(-1).numpy()  # normalized xywh

                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]) - int(xyxy[0]), int(xyxy[3]) - int(xyxy[1]))

                            box_scores.append(f"{round(conf.item(),2)} {c1[0]} {c1[1]} {c2[0]} {c2[1]}")

                            cv2.putText(im0s, '{:.2}'.format(conf.item()), (c1[0],c1[1]-10), cv2.FONT_HERSHEY_SIMPLEX ,1, (255,255,255), 2, cv2.LINE_AA)

#                             cc1, cc2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]) , int(xyxy[3]))

                            cv2.rectangle(im0s, c1,(c2[0]+c1[0], c2[1]+c1[1]),(255,255,255),2)

        

        result={'image_id': image_id, "PredictionString": " ".join(box_scores)} 

        fig = plt.figure(figsize=(50, 50))

        a = fig.add_subplot(1, 4, 1)

        imgplot = plt.imshow(im0s)

  



        # all_path.append(path)

        # final_conf_bbox.append(box_scores)  

        results.append(result)

 

    return results
class opt:

    weights = "../input/20200704-1-yolov5-ricardo/20200704_1.pt"

    img_size = 1024

    conf_thres = 0.45

    iou_thres = 0.45

    augment = True

    merge = True

    device = '0'

    classes=None

    agnostic_nms = True



opt.img_size = check_img_size(opt.img_size)

print(opt)

with torch.no_grad():

    res = detect()
test_df = pd.DataFrame(res, columns=['image_id', 'PredictionString'])

print(test_df)
test_df.to_csv("/kaggle/working/submission.csv",index=False)
!cat submission.csv