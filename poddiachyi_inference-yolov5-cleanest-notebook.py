import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import sys
import cv2
from matplotlib import pyplot as plt

import glob
from tqdm.auto import tqdm
import shutil as sh

sys.path.insert(0, "../input/weightedboxesfusion")
from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion

import warnings
warnings.filterwarnings('ignore')
!cp -r ../input/yolov5trainstable/* ./
from utils.datasets import LoadImages
from utils.utils import scale_coords, non_max_suppression
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

csv_path = '../input/global-wheat-detection/train.csv'
dataset_path = '../input/global-wheat-detection'
model_path = '../input/train-yolov5-simple/trained_models/weights/best_yolov5x_fold0.pt'
# model_path = '../input/train-yolov5-simple/trained_models/weights/last_yolov5x_fold0.pt'
def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_df(path):
    df = pd.read_csv(csv_path)

    bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:,i]

    df.drop(columns=['bbox'], inplace=True)

    df['x_center'] = df['x'] + df['w'] / 2
    df['y_center'] = df['y'] + df['h'] / 2
    df['classes'] = 0

    df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
    
    return df


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def write_bboxes_to_ann(file, bboxes):
    for j in range(len(bboxes)):
        text = ' '.join(bboxes[j])
        file.write(text)
        file.write("\n")
        
        
def process_bboxes(ann_file, table):
    with open(ann_file, 'w+') as f:
        bboxes = table[['classes','x_center','y_center','w','h']].astype(float).values
        bboxes = bboxes / 1024 # because images always 1024 at the beginning and we need to scale bboxes on original image
        bboxes = bboxes.astype(str)
        write_bboxes_to_ann(f, bboxes)
        

def prepare_data_for_training(df):
    index = list(set(df.image_id))
    val_index = index[0 : len(index)//5]
    source = 'train'
    for name, table in tqdm(df.groupby('image_id')):

        if name in val_index:
            phase = 'val2017/'
        else:
            phase = 'train2017/'

        full_labels_path = os.path.join('convertor', phase, 'labels')
        create_folder(full_labels_path)

        ann_file_path = os.path.join(full_labels_path, name + '.txt') # annotation file
        process_bboxes(ann_file_path, table)

        img_folder = os.path.join('convertor', phase, 'images')
        create_folder(img_folder)

        name_with_ext = name + '.jpg'
        img_src = os.path.join(dataset_path, source, name_with_ext)
        img_dst = os.path.join('convertor', phase, 'images', name_with_ext)
        sh.copy(img_src, img_dst)
!rm -r convertor/
df_train = load_df(csv_path)
prepare_data_for_training(df_train)
! cd convertor/ && ls
def load_model(path):
    model = torch.load(path, map_location=device)['model'].float()  # load to FP32
    model.to(device)
    return model
model = load_model(model_path)
model.eval()
def pred_to_bboxes_and_scores(pred, img_shape, org_img_shape):
    bboxes = []
    scores = []
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_shape, det[:, :4], org_img_shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                xywh = torch.tensor(xyxy).view(-1).numpy()  # normalized xywh
                bboxes.append(xywh)
                scores.append(conf)
                
    return np.array(bboxes), np.array(scores)  
def draw_bboxes(img, bboxes, scores):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.7
    color = (255, 0, 0) 
    text_shift = 5
    thickness = 1
    for b,s in zip(bboxes, scores):
        img_show = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 1) 
        img_show = cv2.putText(img_show, '{:.2}'.format(s), (int(b[0]) + text_shift, int(b[1]) + text_shift), font,  
                       fontScale, color, thickness, cv2.LINE_AA)
    
    return img_show
def plot(imgs):
    if len(imgs) == 1:
        plt.figure(figsize=[20, 20])
        plt.imshow(imgs[0])
    else:
        fig = plt.figure(figsize=(30, 30))
        columns = 5
        rows = 2
        for i in range(len(imgs)):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(imgs[i])
    plt.show()
def detect(model, img_path, img_size, conf_thres, iou_thres, show=False, augment=True):
  
    dataset = LoadImages(img_path, img_size=img_size)
    
    imgs_show =  []

    for path, img, img0, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = model(img, augment=augment)[0]
    
        pred = non_max_suppression(pred, conf_thres, iou_thres, merge=True, classes=None, agnostic=True)
    
        bboxes, scores = pred_to_bboxes_and_scores(pred, img.shape[2:], img0.shape)
        
        if show:
            img_show = draw_bboxes(img0, bboxes, scores)
            imgs_show.append(img_show[:,:,::-1])
    
    if show:
        plot(imgs_show)
        
    return bboxes, scores
def run_wbf(boxes, scores, img_size=512, iou_thr=0.5, skip_box_thr=0.5, weights=None):
    labels = [np.zeros(score.shape[0]) for score in scores]
    boxes = [box / img_size for box in boxes]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes * (img_size)
    return boxes, scores, labels
def generate_pseudo_labels(model, source, img_size, conf_thres=0.5, iou_thres=0.9, is_tta=False):
    
    image_files = os.listdir(source)
    
    phase = 'train2017/'
    
    label_folder = os.path.join('convertor', phase, 'labels')
    create_folder(label_folder)
    img_folder = os.path.join('convertor', phase, 'images')
    create_folder(img_folder)
    
    dataset = LoadImages(source, img_size=img_size)
    
    for path, img, img0, _ in dataset:
        img_id = path.split('/')[-1].split('.')[0]
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = model(img, augment=True)[0]
    
        pred = non_max_suppression(pred, conf_thres, iou_thres, merge=True, classes=None, agnostic=True)
    
        boxes, scores = pred_to_bboxes_and_scores(pred, img.shape[2:], img0.shape)
        
        boxes, scores, labels = run_wbf([boxes], [scores], img_size=img_size, iou_thr=0.6, skip_box_thr=0.5)

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        boxes = boxes[scores >= 0.05].astype(np.int32).clip(min=0, max=1023)
        scores = scores[scores >=float(0.05)]
        
        line = ''
        for box in boxes:
            x1, y1, w, h = box
            xc, yc, w, h = (x1+w/2)/1024, (y1+h/2)/1024, w/1024, h/1024
            line += '0 %f %f %f %f\n'%(xc, yc, w, h)
        
        ann_file = os.path.join(label_folder, img_id + '.txt')
        with open(ann_file, 'w+') as f:
            f.write(line)
        
        name_with_ext = img_id + '.jpg'
        img_dst = os.path.join(img_folder, name_with_ext)
        sh.copy(path, img_dst)
# in order to train only on pseudo labels
!rm -r convertor/train2017/
generate_pseudo_labels(model, '../input/global-wheat-detection/test/', img_size=512)
!cd convertor/train2017/labels/ && ls
!python train.py --img 512 --batch 16 --epochs 50 --data ../input/yolostuff/wheat0.yaml --cfg ../input/yolostuff/yolov5x.yaml --name yolov5x_fold0 --weights ../input/train-yolov5-simple/trained_models/weights/best_yolov5x_fold0.pt
# i didn't figure how and when model is saved. it can be save either as path_1 or as path_2 that's why
new_model_path = ''
path_1 = 'runs/exp1_yolov5x_fold0/weights/best_yolov5x_fold0.pt'
path_2 = 'runs/exp0_yolov5x_fold0/weights/best_yolov5x_fold0.pt'
if os.path.exists(path_1):
    new_model_path = path_1
else:
    new_model_path = path_2
new_model = load_model(new_model_path)
new_model.eval()
def format_prediction_string(bboxes, scores):
    pred_strings = []
    for j in zip(scores, bboxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
def detect_for_sub(model, source, img_size, conf_thres=0.7, iou_thres=0.9, is_tta=False, show=False):
    
    results = []
    
    if show:
    
        fig, ax = plt.subplots(5, 2, figsize=(30, 70))
        count = 0
    
    dataset = LoadImages(source, img_size=img_size)
    
    for path, img, img0, _ in dataset:
        img_id = path.split('/')[-1].split('.')[0]
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = model(img, augment=True)[0]
    
        pred = non_max_suppression(pred, conf_thres, iou_thres, merge=True, classes=None, agnostic=True)
    
        boxes, scores = pred_to_bboxes_and_scores(pred, img.shape[2:], img0.shape)
        
        boxes, scores, labels = run_wbf([boxes], [scores], img_size=img_size, iou_thr=0.6, skip_box_thr=0.5)

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        boxes = boxes[scores >= 0.05].astype(np.int32).clip(min=0, max=1023)
        scores = scores[scores >=float(0.05)]
        
        if show:
            for box, score in zip(boxes,scores):
                cv2.rectangle(img0, (box[0], box[1]), (box[2]+box[0], box[3]+box[1]),(220, 0, 0), 2)
                cv2.putText(img0, '%.2f'%(score), (box[0], box[1]),cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255), 2, cv2.LINE_AA)
            ax[count%5][count//5].imshow(img0)
            count += 1
        
        result = {
            'image_id': img_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(result)
        
        
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    
    return test_df
test_data = '../input/global-wheat-detection/test/'
with torch.no_grad():
    test_df = detect_for_sub(new_model, test_data, 512, conf_thres=0.5, iou_thres=0.9, is_tta=False, show=True)
!rm -r *
test_df.to_csv('submission.csv', index=False)
test_df.head()
test_df