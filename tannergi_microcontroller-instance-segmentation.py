# install dependencies: (use cu100 because colab is on CUDA 10.0)

!pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 

!pip install cython pyyaml==5.1

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

import torch, torchvision

torch.__version__

!gcc --version

# opencv is pre-installed on colab
# install detectron2:

!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html
import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()



# import some common libraries

import numpy as np

import cv2

import matplotlib.pyplot as plt



# import some common detectron2 utilities

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog
!wget https://github.com/TannerGilbert/Detectron2-Train-a-Instance-Segmentation-Model/raw/master/microcontroller_segmentation_data.zip

!unzip microcontroller_segmentation_data.zip

!ls
!ls 'Microcontroller Segmentation'
import os

import numpy as np

import json

from detectron2.structures import BoxMode



def get_microcontroller_dicts(directory):

    classes = ['Raspberry_Pi_3', 'Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora']

    dataset_dicts = []

    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:

        json_file = os.path.join(directory, filename)

        with open(json_file) as f:

            img_anns = json.load(f)



        record = {}

        

        filename = os.path.join(directory, img_anns["imagePath"])

        

        record["file_name"] = filename

        record["height"] = 600

        record["width"] = 800

      

        annos = img_anns["shapes"]

        objs = []

        for anno in annos:

            px = [a[0] for a in anno['points']]

            py = [a[1] for a in anno['points']]

            poly = [(x, y) for x, y in zip(px, py)]

            poly = [p for x in poly for p in x]



            obj = {

                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],

                "bbox_mode": BoxMode.XYXY_ABS,

                "segmentation": [poly],

                "category_id": classes.index(anno['label']),

                "iscrowd": 0

            }

            objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts



from detectron2.data import DatasetCatalog, MetadataCatalog

for d in ["train", "test"]:

    DatasetCatalog.register("microcontroller_" + d, lambda d=d: get_microcontroller_dicts('Microcontroller Segmentation/' + d))

    MetadataCatalog.get("microcontroller_" + d).set(thing_classes=['Raspberry_Pi_3', 'Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora'])

microcontroller_metadata = MetadataCatalog.get("microcontroller_train")
import random



dataset_dicts = get_microcontroller_dicts("Microcontroller Segmentation/train")

for d in random.sample(dataset_dicts, 3):

    img = cv2.imread(d["file_name"])

    v = Visualizer(img[:, :, ::-1], metadata=microcontroller_metadata, scale=0.5)

    v = v.draw_dataset_dict(d)

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()
from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("microcontroller_train",)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.MAX_ITER = 1000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)

trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

cfg.DATASETS.TEST = ("microcontroller_test", )

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode

dataset_dicts = get_microcontroller_dicts('Microcontroller Segmentation/test')

for d in random.sample(dataset_dicts, 3):    

    im = cv2.imread(d["file_name"])

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],

                   metadata=microcontroller_metadata, 

                   scale=0.8, 

                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels

    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize = (14, 10))

    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    plt.show()