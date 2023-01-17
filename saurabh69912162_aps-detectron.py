# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install awscli # you'll need this if you want to download images from Open Images (we'll see this later)
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html
# Some basic setup:
# Setup detectron2 logger

import torch, torchvision
torch.__version__
!gcc --version

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() # this logs Detectron2 information such as what the model is doing when it's training



# import some common libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import random


# import some common detectron2 utilities
from detectron2 import model_zoo # a series of pre-trained Detectron2 models: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
from detectron2.engine import DefaultPredictor # a default predictor class to make predictions on an image using a trained model
from detectron2.config import get_cfg # a config of "cfg" in Detectron2 is a series of instructions for building a model
from detectron2.utils.visualizer import Visualizer # a class to help visualize Detectron2 predictions on an image
from detectron2.data import MetadataCatalog # stores information about the model such as what the training/test data is, what the class names are
# Download the trained model
!wget https://storage.googleapis.com/airbnb-amenity-detection-storage/airbnb-amenity-detection/open-images-data/retinanet_model_final/retinanet_model_final.pth 

# Download the train model config (instructions on how the model was built)
!wget https://storage.googleapis.com/airbnb-amenity-detection-storage/airbnb-amenity-detection/open-images-data/retinanet_model_final/retinanet_model_final_config.yaml
# !wget https://raw.githubusercontent.com/mrdbourke/airbnb-amenity-detection/master/custom_images/airbnb-article-cover.jpeg -O demo.jpeg
# img = cv2.imread("./demo.jpeg")
img = cv2.imread("../input/newaps/aps1.jpg")

import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()
fig.set_size_inches(30,30)
plt.imshow(img)
plt.show()
# Target classes with spaces removed
target_classes = ['Bathtub',
 'Bed',
 'Billiard table',
 'Ceiling fan',
 'Coffeemaker',
 'Couch',
 'Countertop',
 'Dishwasher',
 'Fireplace',
 'Fountain',
 'Gas stove',
 'Jacuzzi',
 'Kitchen & dining room table',
 'Microwave oven',
 'Mirror',
 'Oven',
 'Pillow',
 'Porch',
 'Refrigerator',
 'Shower',
 'Sink',
 'Sofa bed',
 'Stairs',
 'Swimming pool',
 'Television',
 'Toilet',
 'Towel',
 'Tree house',
 'Washing machine',
 'Wine rack']


class_dict = {'0':'Bathtub',
 '1':'Bed',
 '2':'Billiard table',
 '3':'Ceiling fan',
 '4':'Coffeemaker',
 '5':'Couch',
 '6':'Countertop',
 '7':'Dishwasher',
 '8':'Fireplace',
 '9':'Fountain',
 '10':'Gas stove',
 '11':'Jacuzzi',
 '12':'Kitchen & dining room table',
 '13':'Microwave oven',
 '14':'Mirror',
 '15':'Oven',
 '16':'Pillow',
 '17':'Porch',
 '18':'Refrigerator',
 '19':'Shower',
 '20':'Sink',
 '21':'Sofa bed',
 '22':'Stairs',
 '23':'Swimming pool',
 '24':'Television',
 '25':'Toilet',
 '26':'Towel',
 '27':'Tree house',
 '28':'Washing machine',
 '29':'Wine rack'}
# Setup a model config file (set of instructions for the model)
cfg = get_cfg() # setup a default config, see: https://detectron2.readthedocs.io/modules/config.html
cfg.merge_from_file("./retinanet_model_final_config.yaml") # merge the config YAML file (a set of instructions on how to build a model)
cfg.MODEL.WEIGHTS = "./retinanet_model_final.pth" # setup the model weights from the fully trained model

# Create a default Detectron2 predictor for making inference
predictor = DefaultPredictor(cfg)



# Make a prediction the example image from above
outputs = predictor(img)


# Number of predicted amenities to draw on the target image
num_amenities = 10

# Set up a visulaizer instance: https://detectron2.readthedocs.io/modules/utils.html#detectron2.utils.visualizer.Visualizer
visualizer = Visualizer(img_rgb=img[:, :, ::-1], # we have to reverse the color order otherwise we'll get blue images (BGR -> RGB)
                        metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=target_classes), # we tell the visualizer what classes we're drawing (from the target classes)
                        scale=0.7)


score_list = outputs["instances"][:num_amenities].to("cpu").scores.tolist() 
class_list = outputs["instances"][:num_amenities].to("cpu").pred_classes.tolist()

for x in range(len(class_list)):
    print(class_dict[str(class_list[x])]+' : '+str(score_list[x]))

# Draw the models predictions on the target image
visualizer = visualizer.draw_instance_predictions(outputs["instances"][:num_amenities].to("cpu"))


fig = plt.figure()
fig.set_size_inches(30,30)
plt.imshow(visualizer.get_image()[:, :, ::-1])
plt.show()

# Display the image
# cv2_imshow(visualizer.get_image()[:, :, ::-1])