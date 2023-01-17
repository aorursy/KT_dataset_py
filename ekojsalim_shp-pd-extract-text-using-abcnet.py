import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!gcc --version

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
import os, sys

!git clone https://github.com/aim-uofa/AdelaiDet.git
os.chdir('AdelaiDet')
DATA_DIR = '/kaggle/input'
ROOT_DIR = '/kaggle/working'

sys.path.append(os.path.join(ROOT_DIR, 'AdelaiDet')) 
!python setup.py build develop
!wget -O tt_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/t2EFYGxNpKPUqhc/download
!ls -lh tt_attn_R_50.pth
import cv2
import glob
import matplotlib.pyplot as plt

def process(filename):
    plt.figure(figsize=(25,15))
    plt.imshow(filename)
z = glob.glob("/kaggle/input/shopee-product-detection-student/train/train/train/00/*.jpg")[:100]
images = [cv2.imread(file) for file in z]
print(len(images))
    
i = 0
for file in images:
    process(file)
    i += 1
    if i > 4: break
os.chdir("demo")
!pwd
import argparse
import multiprocessing as mp
import os
import time
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

from tqdm.auto import tqdm, trange
from tqdm import tqdm_notebook
logger = setup_logger()
cfg = get_cfg()
cfg.merge_from_file("../configs/BAText/TotalText/attn_R_50.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS", "../tt_attn_R_50.pth"])
confidence = 0.5
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence
cfg.freeze()
demo = VisualizationDemo(cfg)
inmages = glob.glob("/kaggle/input/shopee-product-detection-student/train/train/train/**/*.jpg")
def decode_recognition(rec):
    CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

    s = ''
    for c in rec:
        c = int(c)
        if c < 95:
            s += CTLABELS[c]
        elif c == 95:
            s += u'口'
    return s
p = []
for path in tqdm(inmages):
    # use PIL, to be consistent with evaluation
    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    tqdm.write(
        "{}: detected {} instances in {:.2f}s".format(
            path, len(predictions["instances"]), time.time() - start_time
        )
    )
    p.append([decode_recognition(p) for p in predictions["instances"].recs])
anott = pd.DataFrame({'path': inmages, 'annot': p})
anott.to_csv("../../annot_train.csv", index=False)
flatten = lambda l: [item for sublist in l for item in sublist]

c = pd.Series(flatten(anott.annot.values))
c.apply(lambda v: v.lower()).value_counts()[:60]