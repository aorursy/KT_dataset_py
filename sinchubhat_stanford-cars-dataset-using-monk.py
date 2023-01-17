! git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git
# Check version of Cuda
! nvcc -V
! cd Monk_Object_Detection/3_mxrcnn/installation && cat requirements_cuda10.1.txt | xargs -n 1 -L 1 pip install
import matplotlib.pyplot as plt
import cv2
f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_test/cars_test/00001.jpg')
img2 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_test/cars_test/00002.jpg')
img3 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_test/cars_test/00003.jpg')
img4 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_test/cars_test/00004.jpg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)
f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_train/cars_train/00001.jpg')
img2 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_train/cars_train/00002.jpg')
img3 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_train/cars_train/00003.jpg')
img4 = cv2.imread('/kaggle/input/stanford-cars-dataset/cars_train/cars_train/00004.jpg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)
import scipy.io
cars_annos = scipy.io.loadmat('/kaggle/input/stanford-cars-dataset/cars_annos.mat')
cars_annos.keys()
# cars_annos.values()
# annotations
ann = cars_annos['annotations']
# print(ann)
ann.shape
ann.size
class_names = cars_annos['class_names']
# print(class_names)
class_names.shape
class_names.size
print("Annotation: ",ann[0,0])
print("Classname: ",class_names[0,ann[0,0][5]])
row = ann[0,0]
#print(row)
class_path = row[0] # relative_im_path
print(class_path)
x1 = row[1] # bbox_x1
print(x1)
y1 = row[2] # bbox_y1
print(y1)
x2 = row[3] # bbox_x2
print(x2)
y2 = row[4] # bbox_y2
print(y2)
rclass = row[5] # class
print(rclass)
print(class_names[0,rclass])
rtest = row[6] # test
print(rtest)
print(class_names[0,rtest])
row = ann[0,123]
ann_path = row['relative_im_path']
print(str(ann_path))
ann_x1 = row['bbox_x1']
print(int(ann_x1))
ann_x2 = row['bbox_x2']
print(int(ann_x2))
ann_y1 = row['bbox_y1']
print(int(ann_y1))
ann_y2 = row['bbox_y2']
print(int(ann_y2))
ann_class_no = row['class']
print(ann_class_no)
ann_class = class_names[0,ann_class_no-1]
print(str(ann_class))
ann_test_no = row['test']
print(ann_test_no)
ann_test = class_names[0,ann_test_no]
print(str(ann_test))
length = ann.size
import pandas as pd
df_ann = pd.DataFrame(columns = ['relative_im_path','bbox_x1','bbox_y1','bbox_x2','bbox_y2','class','test'])
df_ann
for i in range(length):
    row = ann[0,i]
    df_ann.loc[i,'relative_im_path'] = str("\'") + '/kaggle/input/stanford-cars-dataset/cars_train/cars_train/' + str(row['relative_im_path'])[10:-2] + str("\'")
    df_ann.loc[i,'bbox_x1'] = int(row['bbox_x1'])
    df_ann.loc[i,'bbox_y1'] = int(row['bbox_y1'])
    df_ann.loc[i,'bbox_x2'] = int(row['bbox_x2'])
    df_ann.loc[i,'bbox_y2'] = int(row['bbox_y2'])
    ann_class_no = int(row['class'])
    df_ann.loc[i,'class'] = str(class_names[0,ann_class_no-1])[1:-1].replace(" ", "_")
    ann_test_no = int(row['test'])
    df_ann.loc[i,'test'] = str(class_names[0,ann_test_no])[1:-1].replace(" ", "_")
df_ann.head()
print(df_ann.loc[0,'relative_im_path'])
print(df_ann.loc[0,'class'])
print(df_ann.loc[0,'test'])
df_ann.tail()
df_ann.shape
combined = [];
import numpy as np
for index, row in df_ann.iterrows():
    img_file = str(row['relative_im_path'])[-10:-1];
    #label = str(row['class']).encode("ascii");
    label = str(row['class']);
    x1 = str(row['bbox_x1'])
    x2 = str(row['bbox_x2'])
    y1 = str(row['bbox_y1'])
    y2 = str(row['bbox_y2'])
    wr = "";
    # wr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + label.decode("ascii")[1:-1];
    wr += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + label[1:-1];
    combined.append([img_file, wr]);
combined[:10]
df = pd.DataFrame(combined, columns = ['ID', 'Labels'])  
df.to_csv("/kaggle/working/train_labels.csv", index=False)
import os
import numpy as np 
import cv2
import dicttoxml
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from tqdm import tqdm
import shutil
import json
import pandas as pd
root = "/kaggle";
img_dir = "input/stanford-cars-dataset/cars_train/cars_train/";
anno_file = "working/train_labels.csv";
dataset_path = root;
images_folder = root + "/" + img_dir;
annotations_path = root + "/working/annotations/";
if not os.path.isdir(annotations_path):
    os.mkdir(annotations_path)
    
input_images_folder = images_folder;
input_annotations_path = root + "/" + anno_file;
output_dataset_path = root;
output_image_folder = input_images_folder;
output_annotation_folder = annotations_path;

tmp = img_dir.replace("/", "");
output_annotation_file = output_annotation_folder + "/instances_" + tmp + ".json";
output_classes_file = output_annotation_folder + "/classes.txt";
if not os.path.isdir(output_annotation_folder):
    os.mkdir(output_annotation_folder);
df = pd.read_csv(input_annotations_path);
columns = df.columns
df.head()
df.tail()
columns
delimiter = " ";
list_dict = [];
anno = [];
for i in range(len(df)):
    img_name = df[columns[0]][i];
    labels = df[columns[1]][i];
    tmp = labels.split(delimiter);
    # print(tmp)
    for j in range((len(tmp)//5)):
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
# there are 8144 images in cars_train folder of the dataset
for i in tqdm(range(8144)):
    img_name = df[columns[0]][i];
    labels = df[columns[1]][i];
    tmp = labels.split(delimiter);
    # image_in_path = input_images_folder + img_name;
    image_in_path = root + "/" + img_dir + img_name;
    print(image_in_path)
    image = cv2.imread(image_in_path, 1);
    h, w, c = image.shape;

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
! pwd
! mv /kaggle/working/annotations /kaggle/working/car_dataset
! mkdir /kaggle/working/car_dataset/train   # img_dir
! mkdir /kaggle/working/car_dataset/annotations # anno_dir
! ls /kaggle/working/car_dataset
! mv /kaggle/working/car_dataset/classes.txt /kaggle/working/car_dataset/annotations
! mv /kaggle/working/car_dataset/instances_inputstanford-cars-datasetcars_traincars_train.json /kaggle/working/car_dataset/annotations
# rename
! mv /kaggle/working/car_dataset/annotations/instances_inputstanford-cars-datasetcars_traincars_train.json /kaggle/working/car_dataset/annotations/instances_train.json
! ls /kaggle/working/car_dataset
! ls /kaggle/working/car_dataset/annotations
! cp -r "/kaggle/input/stanford-cars-dataset/cars_train/cars_train/"*.jpg "/kaggle/working/car_dataset/train/"
# ! ls /kaggle/working/car_dataset/train
import os
import sys
sys.path.append("/kaggle/working/Monk_Object_Detection/3_mxrcnn/lib/")
sys.path.append("/kaggle/working/Monk_Object_Detection/3_mxrcnn/lib/mx-rcnn")
from train_base import *
# to be changed accordingly
root_dir = "/kaggle/working";
coco_dir = "car_dataset";
img_dir = "train";
set_dataset_params(root_dir=root_dir, 
                   coco_dir=coco_dir, imageset=img_dir);
set_model_params(model_name="resnet50");
set_hyper_params(gpus="0", lr=0.001, lr_decay_epoch="1", epochs=10, batch_size=16);
set_output_params(log_interval=100, save_prefix="model_resnet50");
set_img_preproc_params(img_short_side=600, img_long_side=1000, 
                       mean=(123.68, 116.779, 103.939), std=(1.0, 1.0, 1.0));
initialize_rpn_params();
initialize_rcnn_params();
if os.path.isdir("./cache/"):
    os.system("rm -r ./cache/")
roidb = set_dataset();
sym = set_network();
train(sym, roidb);
from infer_base import *
class_file = set_class_list("/kaggle/working/car_dataset/annotations/classes.txt");
set_model_params(model_name="resnet50", model_path="trained_model/model_resnet50-0006.params");
set_hyper_params(gpus="0", batch_size=1);
set_img_preproc_params(img_short_side=600, img_long_side=1000, 
                       mean=(123.68, 116.779, 103.939), std=(1.0, 1.0, 1.0));
initialize_rpn_params();
initialize_rcnn_params();
sym = set_network();
mod = load_model(sym);
set_output_params(vis_thresh=0.6, vis=True)
Infer("/kaggle/input/stanford-cars-dataset/cars_test/cars_test/00004.jpg", mod);