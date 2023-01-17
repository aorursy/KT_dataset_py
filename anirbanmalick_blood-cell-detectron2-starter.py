!/opt/conda/bin/python3.7 -m pip install --upgrade pip
# install dependencies: 

#!pip install pycocotools>=2.0.1

import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

#!gcc --version

# opencv is pre-installed on colab
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
import os

import numpy as np

import pandas as pd

import cv2 as cv

import matplotlib.pyplot as plt
root = '/kaggle/input/blood-cell-count-and-typesdetection/'
random_image = os.listdir(root+'images/images/train')[1]

image = cv.imread(root+'images/images/train/'+random_image)

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

plt.imshow(image)

plt.show()
#look at yaml file

import yaml

a_yaml_file = open("/kaggle/input/blood-cell-count-and-typesdetection/bcc-kaggle.yaml")

parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)



print(parsed_yaml_file)
img = cv.imread('/kaggle/input/blood-cell-count-and-typesdetection/images/images/train/BloodImage_00372.jpg')

plt.imshow(img)
from detectron2.structures import BoxMode



label_folder = root + 'labels/labels/'

image_folder = root + 'images/images/'



def prepare_dataset(path,image_folder=image_folder,label_folder=label_folder):

    list_dataset = []

    files = os.listdir(label_folder+path)

    print("Number of lables in", path, str(len(files)))

    for i,file in enumerate(files):

        try:

            all_vals = []

            with open(label_folder+path+file,'r') as f:

                all_vals.append(f.readlines())

                f.close()

            all_vals = [line.split() for line in all_vals[0]]

            master_dict = {}

            annotations = []

            for i,vals in enumerate(all_vals):

                image_id = file[:-4]

                label_file_name = label_folder+path+file

                image_file_name= image_folder+path+file

                image_file_name = str(image_file_name).replace('.txt','.jpg')

                height, width = cv.imread(image_file_name).shape[:2]

                label,x,y,w,h= [float(i) for i in vals]

                x,w = x*width,w*width

                y,h = y*height,h*height

    #             x_min,y_min,x_max,y_max = x,y,x+w,y+h

                x_min,y_min,x_max,y_max = int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)

                bbox = [x_min,y_min,x_max,y_max]



                obj = {

                    "bbox": [x_min, y_min,x_max, y_max],

                    "bbox_mode": BoxMode.XYXY_ABS,

                    "category_id": label,

                }

                annotations.append(obj)

                master_dict['file_name']=image_file_name

                master_dict['image_id'] = image_id

                master_dict['height'] = height

                master_dict['width'] = width

        except:

            pass

        master_dict['annotations'] = annotations

        list_dataset.append(master_dict)

    return list_dataset





def prepare_master_dict(root,path='valid/'):



    unique_image_id_train = list(set(os.listdir(image_folder+path)))

    unique_image_id_train = [i[:-4] for i in unique_image_id_train]

    

    list_dataset = prepare_dataset(path=path)

    return list_dataset
list_dataset = prepare_master_dict(root=root)

# list_dataset
from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog
for d in ["train/", "valid/"]:

    DatasetCatalog.register("bc_v6_" + d, lambda d=d: prepare_master_dict(root=root,path=d))

    MetadataCatalog.get("bc_v6_" + d).set(thing_classes=['Platelets', 'RBC', 'WBC'])
# with open('/kaggle/input/blood-cell-count-and-typesdetection/labels/labels/train/BloodImage_00172.txt','r') as f:

#     print(f.readlines())
# img = cv.imread('/kaggle/input/blood-cell-count-and-typesdetection/images/images/train/BloodImage_00172.jpg')

# plt.imshow(img)
import random

valid_metadata = MetadataCatalog.get("bc_v5_" + 'valid')

for d in random.sample(list_dataset, 1):

    print(d['file_name'])

    img = cv.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], metadata=valid_metadata, scale=0.5)

    out = visualizer.draw_dataset_dict(d)

    plt.imshow(out.get_image()[:, :, ::-1])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg

from detectron2.utils.logger import setup_logger

setup_logger()



model_yaml = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(model_yaml))

cfg.DATASETS.TRAIN = ("bc_v6_train/",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)  

cfg.SOLVER.IMS_PER_BATCH = 4

cfg.SOLVER.BASE_LR = 0.003  

cfg.SOLVER.MAX_ITER = 100  

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8   

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   

cfg.DATASETS.TEST = ("bc_v6_valid/", )

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode

dataset_dicts = prepare_master_dict(root=root,path='valid/')



for d in random.sample(dataset_dicts, 3):    

    im = cv.imread(d["file_name"])

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],

                   metadata=valid_metadata, 

                   scale=0.5, 

    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.imshow(v.get_image()[:,:,::-1])

    plt.show()