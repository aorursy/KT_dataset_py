!cp -rf ../input/tttttt/yolov5-master/* .
!mkdir svhn
!mkdir svhn/images
!mkdir svhn/labels
import os
import shutil
train_image_path = '/kaggle/input/streetclassify/input/input/train/'
val_image_path = '/kaggle/input/streetclassify/input/input/val/'

dst_image_path = '/kaggle/working/svhn/images/'
train_image_list = os.listdir(train_image_path)
val_image_list = os.listdir(val_image_path)
for img in train_image_list:
    shutil.copy(train_image_path+img, dst_image_path+img)
for img in val_image_list:
    shutil.copy(val_image_path+img, dst_image_path+'val_'+img)
import os
import cv2
import json
train_image_path = '/kaggle/input/streetclassify/input/input/train/'
val_image_path = '/kaggle/input/streetclassify/input/input/val/'
train_annotation_path = '/kaggle/input/streetclassify/input/input/train.json'
val_annotation_path = '/kaggle/input/streetclassify/input/input/val.json'

train_data = json.load(open(train_annotation_path))
val_data = json.load(open(val_annotation_path))
label_path = '/kaggle/working/svhn/labels/'
for key in train_data:
    f = open(label_path+key.replace('.png', '.txt'), 'w')
    img = cv2.imread(train_image_path+key)
    shape = img.shape
    label = train_data[key]['label']
    left = train_data[key]['left']
    top = train_data[key]['top']
    height = train_data[key]['height']
    width = train_data[key]['width']
    for i in range(len(label)):
        x_center = 1.0 * (left[i]+width[i]/2) / shape[1]
        y_center = 1.0 * (top[i]+height[i]/2) / shape[0]
        w = 1.0 * width[i] / shape[1]
        h = 1.0 * height[i] / shape[0]
        # label, x_center, y_center, w, h
        f.write(str(label[i]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')
    f.close()
for key in val_data:
    f = open(label_path+'val_'+key.replace('.png', '.txt'), 'w')
    img = cv2.imread(val_image_path+key)
    shape = img.shape
    label = val_data[key]['label']
    left = val_data[key]['left']
    top = val_data[key]['top']
    height = val_data[key]['height']
    width = val_data[key]['width']
    for i in range(len(label)):
        x_center = 1.0 * (left[i]+width[i]/2) / shape[1]
        y_center = 1.0 * (top[i]+height[i]/2) / shape[0]
        w = 1.0 * width[i] / shape[1]
        h = 1.0 * height[i] / shape[0]
        # label, x_center, y_center, w, h
        f.write(str(label[i]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')
    f.close()
os.remove(dst_image_path + '012667.png')
os.remove(label_path + '012667.txt')
os.remove(dst_image_path + 'val_003191.png')
os.remove(label_path + 'val_003191.txt')

print(len(os.listdir(dst_image_path)))
print(len(os.listdir(dst_image_path)))
import yaml

aproject = {'train': '../svhn/images/',
            'val': '../svhn/images/',
            'nc': 10,
            'names':['1','2','3','4','5','6','7','8','9','10']
            }
f = open('/kaggle/working/yolov5/data/svhn.yaml','w')
print(yaml.dump(aproject,f))
%cd /kaggle/working/yolov5
!python train.py --img 320 --batch 32 --epochs 1 --data svhn.yaml --cfg yolov5x.yaml --weights yolov5x.pt
%cd /kaggle/working
!zip -r svhn.zip svhn
#!rm -rf  dirWithFiles/*
!rm -rf  /kaggle/working/svhn
!rm -rf  /kaggle/working/yolov5
!rm -rf  /kaggle/working/best.pt