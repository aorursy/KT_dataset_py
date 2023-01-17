%cd /opt/conda/bin/

!python3.7 -m pip install --upgrade pip
! conda install -y gdown
# install dependencies: (use cu101 because colab has CUDA 10.1)

!pip install --use-feature=2020-resolver -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 

!pip install cython pyyaml==5.1

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

!gcc --version

!pip install google-colab
# opencv is pre-installed on colab

# install detectron2:

!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
# You may need to restart your runtime prior to this, to let your installation take effect

# Some basic setup:

# Setup detectron2 logger

import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()



# import some common libraries

import numpy as np

import cv2

import random

from google.colab.patches import cv2_imshow



# import some common detectron2 utilities

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog

from detectron2.data.catalog import DatasetCatalog
import os

output="/kaggle/working/detectron2"

if not os.path.exists(output):

    os.makedirs(output)



%cd  /kaggle/working/detectron2/

# download ataset dataset_21_2825_1

#!gdown --id  1-2uin6c7rTqGoygEZMPvAsLwz18sfPHp

#!tar xvzf dataset_21_2825_1.tar.gz

#!rm -rf dataset_21_2825_1.tar.gz





# download ataset dataset_21_6426

!gdown --id  12V7svOlQRNk88ahJ-mqUP3buuX_rr-J-

!tar xzf dataset_21_6426.tar.gz

!rm -rf dataset_21_6426.tar.gz
import os

output="/kaggle/working/detectron2/output"

if not os.path.exists(output):

    os.makedirs(output)

    

%cd  /kaggle/working/detectron2/output

# download ataset model_final.pt (less accuracy)

!gdown --id  1-XGlG6XKDJCZJyRyJoaB6hQAUZIPXkoP





# download ataset model_final.pt (more accuracy)

# %cp -arvf /kaggle/input/lastpretrainedmodel/model_final.pth /kaggle/working/detectron2/output/model_final.pth
"""

import os

output="/kaggle/working/detectron2"

if not os.path.exists(output):

    os.makedirs(output)

    

%cd /kaggle/working/detectron2





import os

output="dataset_21_2825_1"

if not os.path.exists(output):

    os.makedirs(output)

    

%cd dataset_21_2825_1/

!curl -L "https://app.roboflow.ai/ds/ypwjrTOS9q?key=DD9xOpsn13" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

"""
"""

import fileinput

import os

from shutil import copyfile

def convert_list_to_string(org_list, seperator=' '):

  #Convert list to string, by joining all item in list with given separator.

  #Returns the concatenated string 

  return seperator.join(org_list)



listOfFiles=["test","train","valid"]

input_dir="/kaggle/working/detectron2/dataset_21_2825_1"

for folderName in listOfFiles:

  output=os.path.join("/content",folderName)

  if not os.path.exists(output):

    os.makedirs(output)



  input_file=os.path.join(input_dir,folderName,"_annotations.coco.json")

  

  copyfile(input_file, os.path.join(output,"_annotations.coco.json"))

  for i in range(0,21):

    text_to_search='"category_id": {},'.format(i+1)

    # print(text_to_search)

    replacement_text='"category_id": {},'.format(i)

    #read input file

    fin = open(input_file, "rt")

    #read file contents to string

    data = fin.read()

    #replace all occurrences of the required string

    data = data.replace(text_to_search, replacement_text)

    #close the input file

    fin.close()

    #open the input file in write mode

    fin = open(input_file, "wt")

    #overrite the input file with the resulting data

    fin.write(data)

    #close the file

    fin.close()

    print("\rpercent {:.2f}%".format(100*i/20), end='')



"""



%cd /kaggle/working/detectron2/



from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train", {}, "dataset_21_6426/train/_annotations.coco.json", "dataset_21_6426/train")

register_coco_instances("my_dataset_val", {}, "dataset_21_6426/valid/_annotations.coco.json", "dataset_21_6426/valid")

register_coco_instances("my_dataset_test", {}, "dataset_21_6426/test/_annotations.coco.json", "dataset_21_6426/test")

print("visualize training data")

# #visualize training data

# my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")

# dataset_dicts = DatasetCatalog.get("my_dataset_train")



# import random

# from detectron2.utils.visualizer import Visualizer



# for d in random.sample(dataset_dicts, 5):

#     img = cv2.imread(d["file_name"])



#     print(d["file_name"].split("/")[-1].split("_jpg")[0])

#     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.7)

#     vis = visualizer.draw_dataset_dict(d)

#     cv2_imshow(vis.get_image()[:, :, ::-1])

#We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.



from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator



class CocoTrainer(DefaultTrainer):



  @classmethod

  def build_evaluator(cls, cfg, dataset_name, output_folder=None):



    if output_folder is None:

        os.makedirs("coco_eval", exist_ok=True)

        output_folder = "coco_eval"



    return COCOEvaluator(dataset_name, cfg, False, output_folder)
%cd /kaggle/working/detectron2/

#from .detectron2.tools.train_net import Trainer

#from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg

#from detectron2.evaluation.coco_evaluation import COCOEvaluator

import os



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("my_dataset_train",)

cfg.DATASETS.TEST = ("my_dataset_val",)



cfg.DATALOADER.NUM_WORKERS = 4



# use coco pre-trained model

#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo



# use my pre-trained model

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")



cfg.SOLVER.IMS_PER_BATCH =   1



cfg.SOLVER.BASE_LR = 0.001





#cfg.SOLVER.WARMUP_ITERS = 10000

cfg.SOLVER.MAX_ITER = 460000 #adjust up if val mAP is still rising, adjust down if overfit

#cfg.SOLVER.STEPS = (10000, 10500)

cfg.SOLVER.GAMMA = 0.05









cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 7

#             batch_size_per_image (int): number of proposals to sample for training

#             positive_fraction (float): fraction of positive (foreground) proposals

#                 to sample for training.

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21 #your number of classes + 1

# num_classes (int): number of classes. Used to label background proposals.



cfg.TEST.EVAL_PERIOD = 1000





os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)

trainer.resume_or_load(resume=True)

    # resume_or_load(resume=True)[source]

    # If resume==True, and last checkpoint exists, resume from it, load all checkpointables (eg. optimizer and scheduler) and update iteration counter from it. cfg.MODEL.WEIGHTS will not be used.



    # Otherwise, load the model specified by the config (skip all checkpointables) and start from the first iteration.

trainer.train()
#%cp -arvf /kaggle/working/detectron2/output/model_final.pth /kaggle/working/
# Look at training curves in tensorboard:

# !kill 669

%load_ext tensorboard

%tensorboard --logdir output
#test evaluation

import os

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

from detectron2.evaluation import COCOEvaluator, inference_on_dataset



cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001 # set the testing threshold for this model

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")

val_loader = build_detection_test_loader(cfg, "my_dataset_test")

inference_on_dataset(trainer.model, val_loader, evaluator)
%ls -l /kaggle/working/detectron2/output/
%cd /kaggle/working/detectron2/

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.DATASETS.TEST = ("my_dataset_test", )

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001   # set the testing threshold for this model

predictor = DefaultPredictor(cfg)

test_metadata = MetadataCatalog.get("my_dataset_test")
%cd /kaggle/working/detectron2/



from detectron2.utils.visualizer import ColorMode

import glob



listOfImages=glob.glob('dataset_21_6426/test/*jpg')

for i in range(0,5):

  im = cv2.imread(listOfImages[i])

  outputs = predictor(im)

  v = Visualizer(im[:, :, ::-1],

                metadata=test_metadata, 

                scale=1

                 )

  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

  cv2_imshow(out.get_image()[:, :, ::-1])

%cd /kaggle/working/

# download the test image

!gdown --id 1-UxNH8cM0km391pO7-JAo2qX8TNKWmJV
%cd /kaggle/working/detectron2/



from detectron2.utils.visualizer import ColorMode

import glob

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.DATASETS.TEST = ("my_dataset_test", )



for x in [0.0001,0.001,0.0015,0.009,0.01,0.1]:

  print("\n\t\tSCORE_THRESH_TEST={}\n".format(x))

  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = x

  # set the testing threshold for this model

  predictor = DefaultPredictor(cfg)

  W=1024

  imageName="/kaggle/working/0.jpg"

  im = cv2.imread(imageName)

  im=cv2.resize(im,(W,W))

  outputs = predictor(im)

  v = Visualizer(im[:, :, ::-1],

                metadata=test_metadata, 

                scale=1

                 )

  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

  cv2_imshow(out.get_image()[:, :, ::-1])