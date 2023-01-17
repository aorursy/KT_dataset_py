!git clone 'https://github.com/bala-codes/Yolo-V5_Object_Detection_Blood_Cell_Count_and_Detection.git' # For dataset downloading
!git clone  'https://github.com/ultralytics/yolov5.git' # For loading official yolo-v5
import shutil
import os, sys, random
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import patches
import numpy as np
import os
shutil.copyfile('/kaggle/working/Yolo-V5_Object_Detection_Blood_Cell_Count_and_Detection/codes/bcc-kaggle.yaml', '/kaggle/working/yolov5/bcc-kaggle.yaml')

!pip install --upgrade pip
!pip install Cython
!pip install matplotlib>=3.2.2
!pip install numpy>=1.18.5
!pip install opencv-python>=4.1.2
!pip install pillow
!pip install PyYAML>=5.3
!pip install scipy>=1.4.1
!pip install tensorboard>=2.2
!pip install tqdm>=4.41.0
!pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__, torchvision.__version__)
os.getcwd()
%%time
# Here I have provided only one epoch, feel free to increment it !!!

!python yolov5/train.py --img 640 --batch 8 --epochs 1 --data yolov5/bcc-kaggle.yaml --cfg /kaggle/working/Yolo-V5_Object_Detection_Blood_Cell_Count_and_Detection/Inference_files/models/yolov5s.yaml --name BCCM
# Detection or Run Inference on the model
!python yolov5/detect.py --source /kaggle/working/Yolo-V5_Object_Detection_Blood_Cell_Count_and_Detection/dataset-preprocessed/bcc/images/valid/ --weights /kaggle/working/Yolo-V5_Object_Detection_Blood_Cell_Count_and_Detection/Inference_files/best_BCCM.pt
disp_images = glob('/kaggle/working/inference/output/*')
fig=plt.figure(figsize=(20, 28))
columns = 3
rows = 5
for i in range(1, columns*rows +1):
    img = np.random.choice(disp_images)
    img = plt.imread(img)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()