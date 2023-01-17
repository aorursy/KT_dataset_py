import os

import sys

os.listdir("/kaggle/input/kaggle-monk-efficientdet/kaggle_monk_efficientdet")
! cp -r /kaggle/input/kaggle-monk-efficientdet/kaggle_monk_efficientdet .
! cd kaggle_monk_efficientdet/cocoapi/PythonAPI/ && pip install -e .
sys.path.append("kaggle_monk_efficientdet/cocoapi/PythonAPI/")
! cd kaggle_monk_efficientdet/EfficientNet-PyTorch/ && pip install -e .
sys.path.append("kaggle_monk_efficientdet/EfficientNet-PyTorch/")
! cd kaggle_monk_efficientdet/ && pip install xmltodict-0.12.0-py2.py3-none-any.whl
! cd kaggle_monk_efficientdet/dicttoxml-1.7.4/ && pip install -e .
sys.path.append("kaggle_monk_efficientdet/dicttoxml-1.7.4/")
import os

import sys

import pandas as pd

import cv2

import numpy as np
os.listdir("/kaggle/input/global-wheat-detection")
df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv");



combined = [];

img_name_current = df.iloc[0]["image_id"] + ".jpg";

wr = "";

from tqdm import tqdm

for i in tqdm(range(len(df))):

    img_name = df.iloc[i]["image_id"] + ".jpg";

    if(img_name_current != img_name):

        wr = wr[:len(wr)-1];

        combined.append([img_name_current, wr]);

        img_name_current = img_name;

        wr = "";

        #break;

    bbox = df.iloc[i]["bbox"]

    bbox = bbox[1:len(bbox)-1].split(",");

    x1 = int(float(bbox[0]));

    y1 = int(float(bbox[1]));

    x2 = int(float(bbox[0])) + int(float(bbox[2]));

    y2 = int(float(bbox[1])) + int(float(bbox[3]));

    

    wr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " wheat ";
df = pd.DataFrame(combined, columns = ['Id', 'Labels']);

df.to_csv("train_labels.csv", index=False)
import os

import numpy as np 

import cv2

from tqdm import tqdm

import shutil

import json

import pandas as pd
os.mkdir("wheat_dataset")
! cp -r /kaggle/input/global-wheat-detection/train wheat_dataset/
root = "wheat_dataset/";

img_dir = "train/";

anno_file = "train_labels.csv";







dataset_path = root;

images_folder = root + "/" + img_dir;

annotations_path = "/annotations/";



if not os.path.isdir(annotations_path):

    os.mkdir(annotations_path)

    

input_images_folder = images_folder;

input_annotations_path = anno_file;



output_dataset_path = root;

output_image_folder = input_images_folder;

output_annotation_folder = root + "/" + annotations_path;



tmp = img_dir.replace("/", "");

output_annotation_file = output_annotation_folder + "/instances_" + tmp + ".json";

output_classes_file = output_annotation_folder + "/classes.txt";



if not os.path.isdir(output_annotation_folder):

    os.mkdir(output_annotation_folder);

    

df = pd.read_csv(input_annotations_path);

columns = df.columns



delimiter = " ";



list_dict = [];

anno = [];

for i in range(len(df)):

    img_name = df[columns[0]][i];

    labels = df[columns[1]][i];

    tmp = labels.split(delimiter);

    for j in range(len(tmp)//5):

        label = tmp[j*5+4];

        if(label not in anno):

            anno.append(label);

    anno = sorted(anno)

    

for i in tqdm(range(len(anno))):

    tmp = {};

    tmp["supercategory"] = "master";

    tmp["id"] = i;

    tmp["name"] = anno[i];

    list_dict.append(tmp);



anno_f = open(output_classes_file, 'w');

for i in range(len(anno)):

    anno_f.write(anno[i] + "\n");

anno_f.close();



coco_data = {};

coco_data["type"] = "instances";

coco_data["images"] = [];

coco_data["annotations"] = [];

coco_data["categories"] = list_dict;

image_id = 0;

annotation_id = 0;





for i in tqdm(range(len(df))):

    img_name = df[columns[0]][i];

    labels = df[columns[1]][i];

    tmp = labels.split(delimiter);

    image_in_path = input_images_folder + "/" + img_name;

    img = cv2.imread(image_in_path, 1);

    h, w, c = img.shape;



    images_tmp = {};

    images_tmp["file_name"] = img_name;

    images_tmp["height"] = h;

    images_tmp["width"] = w;

    images_tmp["id"] = image_id;

    coco_data["images"].append(images_tmp);

    



    for j in range(len(tmp)//5):

        x1 = int(tmp[j*5+0]);

        y1 = int(tmp[j*5+1]);

        x2 = int(tmp[j*5+2]);

        y2 = int(tmp[j*5+3]);

        label = tmp[j*5+4];

        annotations_tmp = {};

        annotations_tmp["id"] = annotation_id;

        annotation_id += 1;

        annotations_tmp["image_id"] = image_id;

        annotations_tmp["segmentation"] = [];

        annotations_tmp["ignore"] = 0;

        annotations_tmp["area"] = (x2-x1)*(y2-y1);

        annotations_tmp["iscrowd"] = 0;

        annotations_tmp["bbox"] = [x1, y1, x2-x1, y2-y1];

        annotations_tmp["category_id"] = anno.index(label);



        coco_data["annotations"].append(annotations_tmp)

    image_id += 1;



outfile =  open(output_annotation_file, 'w');

json_str = json.dumps(coco_data, indent=4);

outfile.write(json_str);

outfile.close();
import os

import sys

sys.path.append("kaggle_monk_efficientdet/Monk_Object_Detection/4_efficientdet/lib/");
from train_detector import Detector

gtf = Detector();
root_dir = "wheat_dataset/";

coco_dir = "";

img_dir = "";

set_dir = "train";
gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=8, image_size=512, use_gpu=True)
! mkdir /root/.cache/torch
! mkdir /root/.cache/torch/checkpoints
! cp kaggle_monk_efficientdet/efficientnet-b0-355c32eb.pth /root/.cache/torch/checkpoints/efficientnet-b0-355c32eb.pth
gtf.Model();
gtf.Set_Hyperparams(lr=0.0001, val_interval=1, es_min_delta=0.0, es_patience=0)
gtf.Train(num_epochs=50, model_output_dir="trained/");
import os

import sys

sys.path.append("Monk_Object_Detection/4_efficientdet/lib/");
from infer_detector import Infer
gtf = Infer();
gtf.Model(model_dir="trained/")
f = open("wheat_dataset/annotations/classes.txt", 'r');

class_list = f.readlines();

f.close();

for i in range(len(class_list)):

    class_list[i] = class_list[i][:-1]
img_list = os.listdir("/kaggle/input/global-wheat-detection/test/")
combined = [];

for i in tqdm(range(len(img_list))):

    img_id = img_list[i].split(".")[0];

    img_path = "/kaggle/input/global-wheat-detection/test/" + img_list[i];

    scores, labels, boxes = gtf.Predict(img_path, class_list, vis_threshold=0.4);

    #print(scores.shape, boxes.shape);

    wr = "";

    for j in range(scores.shape[0]):

        score = float(scores[j].cpu().numpy())

        if(score > 0.4):

            x1 = int(boxes[j].cpu().numpy()[0]);

            y1 = int(boxes[j].cpu().numpy()[1]);

            x2 = int(boxes[j].cpu().numpy()[2]);

            y2 = int(boxes[j].cpu().numpy()[3]);

            w = x2-x1;

            h = y2-y1;

            wr += str(score) + " " + str(x1) + " " + str(y1) + " " + str(w) + " " + str(h) + " ";

        

    wr = wr[:len(wr)-1];

    combined.append([img_id, wr]);
df = pd.DataFrame(combined, columns = ['image_id', 'PredictionString']);

df.to_csv("submission.csv", index=False)
! rm -r wheat_dataset
! rm -r kaggle_monk_efficientdet