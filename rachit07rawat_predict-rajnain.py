import numpy as np 

import pandas as pd 

import os



train_path = '../input/global-wheat-detection/train.csv'

df = pd.read_csv(train_path)



bbox = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i,col_name in enumerate(['x','y','w','h']):

    df[col_name] = bbox[:,i]

df['center_x'] = df['x'] + df['w']/2

df['center_y'] = df['y'] + df['h']/2

df['classes'] = 0



df = df[['image_id','x', 'y', 'w', 'h','center_x','center_y','classes']]
!cp -r ../input/yolov5train/* .
from utils.datasets import *

from utils.utils import * 
import sys

sys.path.insert(0, "../input/weightedboxesfusion")
class opt:

        weights = "../input/wheatyolo/best_wheat.pt"

        img_size = 1024

        conf_thres = 0.3

        iou_thres = 0.4

        augment = True

        device = '0'

        classes=None

        agnostic_nms = True



def detect(save_img=False):       

    weights, imgsz = opt.weights,opt.img_size

    source = '../input/global-wheat-detection/test/'

    

    # Initialize

    device = torch_utils.select_device(opt.device)

    half = False       

    

    model = torch.load(weights, map_location=device)['model'].to(device).float().eval()

    

    dataset = LoadImages(source, img_size=1024)



    t0 = time.time()



    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img



    all_path=[]

    all_bboxex =[]

    all_score =[]

    for path, img, im0s, vid_cap in dataset:

            print(im0s.shape)

            img = torch.from_numpy(img).to(device)

            img = img.half() if half else img.float()  # uint8 to fp16/32

            img /= 255.0  # 0 - 255 to 0.0 - 1.0



            if img.ndimension() == 3:

                img = img.unsqueeze(0)



            # Inference

            t1 = torch_utils.time_synchronized()

            bboxes_2 = []

            score_2 = []        



            if True:

                pred = model(img, augment=opt.augment)[0]

                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False)

                t2 = torch_utils.time_synchronized()



                bboxes = []

                score = []



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

                                xywh = torch.tensor(xyxy).view(-1).numpy()  



                                bboxes.append(xywh)

                                score.append(conf)    

                bboxes_2.append(bboxes)

                score_2.append(score)

            all_path.append(path)

            all_score.append(score_2)

            all_bboxex.append(bboxes_2)    

    return all_path,all_score,all_bboxex

            

#opt.img_size = check_img_size(opt.img_size)



with torch.no_grad():

    res = detect()

all_path,all_score,all_bboxex = res  
def run_wbf(boxes,scores, image_size=1024, iou_thr=0.6, skip_box_thr=0.24, weights=None):

#     boxes =boxes/(image_size-1)

    

    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    

    #     boxes = boxes*(image_size-1)

#     boxes = boxes

    

    return boxes, scores, labels
#!pip install ensemble-boxes

from ensemble_boxes import *
results =[]



def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)



for row in range(len(all_path)):

    image_id = all_path[row].split("/")[-1].split(".")[0]

    boxes = all_bboxex[row]

    scores = all_score[row]

    print(type(boxes))

    boxes, scores, labels = run_wbf(boxes,scores)

    boxes = np.array(boxes)

    print(boxes.shape)

    boxes = (boxes*1024/1024).astype(np.int32).clip(min=0, max=1023)

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}

    results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.shape
print(test_df.head(10))
test_df.to_csv('submission.csv', index = False)