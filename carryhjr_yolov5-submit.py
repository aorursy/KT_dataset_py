half = False
import sys

sys.path.insert(0, '/kaggle/input/yolov5/yolov5/')
import torch

device = torch.device('cuda:0')

model = torch.load('/kaggle/input/wheat-submit/best_wheat1024.pt', map_location=device)['model'].to(device).float().eval()

if half:

    model.half()
import glob

img_paths = glob.glob('/kaggle/input/global-wheat-detection/test/*.jpg')

print(img_paths)
def inference_detector(model, img_path):

    from utils.datasets import LoadImages

    dataset = LoadImages(img_path, img_size=1024)

    path, img, im0, vid_cap = next(iter(dataset))

    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32

    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:

        img = img.unsqueeze(0)

    pred = model(img, augment=True)[0]

    from utils.utils import non_max_suppression

    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=True)

    from utils.utils import scale_coords

    bboxes = []

    scores = []

    clses = []

    for i, det in enumerate(pred):  # detections per image

        if det is not None and len(det):

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in det:

                xyxy = torch.tensor(xyxy).view(-1).numpy()

                bboxes.append([*xyxy, conf.item()])

    return np.array(bboxes)



import sys

sys.path.insert(0, "../input/weightedboxesfusion")

from ensemble_boxes import *

def run_wbf(boxes,scores, iou_thr=0.4, skip_box_thr=0.34, weights=None):

    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes, scores, labels





def inference_detector_wbf(model, img_path):

    from utils.datasets import LoadImages

    dataset = LoadImages(img_path, img_size=640)

    path, img, im0, vid_cap = next(iter(dataset))

    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32

    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:

        img = img.unsqueeze(0)

    pred = model(img, augment=True)[0]

    from utils.utils import non_max_suppression

    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.95, classes=None, agnostic=False)

    from utils.utils import scale_coords

    bboxes = []

    scores = []

    clses = []

    for i, det in enumerate(pred):  # detections per image

        if det is not None and len(det):

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in det:

                xyxy = torch.tensor(xyxy).view(-1).numpy()

                bboxes.append(xyxy)

                scores.append(conf.item())

    bboxes = [bboxes]

    scores = [scores]

    bboxes, scores, labels = run_wbf(bboxes, scores)

    det = []

    for bbox, score in zip(bboxes, scores):

        det.append([*bbox, score])

    return np.array(det)
# test

import numpy as np

import cv2

def vis(image_path, det):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    size = 300

    idx = -1

    font = cv2.FONT_HERSHEY_SIMPLEX 

    # fontScale 

    fontScale = 1

    # Blue color in BGR 

    color = (255, 0, 0) 

    bboxes = det[:,:4].astype(np.int32)

    scores = det[:,4]

    # Line thickness of 2 px 

    thickness = 2

    for b,s in zip(bboxes,scores):

        if s > 0.1:

            image = cv2.rectangle(image, (b[0],b[1]), (b[2],b[3]), (255,0,0), 1) 

            image = cv2.putText(image, '{:.2}'.format(s), (b[0]+np.random.randint(20),b[1]), font,  

                           fontScale, color, thickness, cv2.LINE_AA)

    import matplotlib.pyplot as plt

    plt.figure(figsize=[6, 6])

    plt.imshow(image[:,:,::-1])

    plt.show()

import glob

img_paths = glob.glob('/kaggle/input/global-wheat-detection/test/*.jpg')

img_path = img_paths[0]

det = inference_detector_wbf(model, img_path)

vis(img_path, det)
import numpy as np

img_paths = glob.glob('/kaggle/input/global-wheat-detection/test/*.jpg')

results = []

from tqdm import tqdm

for img_path in tqdm(img_paths):

    det = inference_detector_wbf(model, img_path)

    pred_strings = []

    for bbox in det:

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(bbox[4], bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]))

    pred_strings = " ".join(pred_strings)

    result = {'image_id': img_path.split('/')[-1].split('.')[0], 'PredictionString': pred_strings}

    results.append(result)

import pandas as pd

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)