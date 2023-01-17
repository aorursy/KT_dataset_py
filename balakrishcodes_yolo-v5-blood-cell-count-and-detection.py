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
!git clone  'https://github.com/ultralytics/yolov5.git' # For loading official yolo-v5
import os, sys, random, shutil
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import patches
import numpy as np
import os
shutil.copyfile('/kaggle/input/blood-cell-count-and-typesdetection/bcc-kaggle.yaml', '/kaggle/working/yolov5/bcc-kaggle.yaml')
!cp -r '/kaggle/input/blood-cell-count-and-typesdetection' '/kaggle/working/blood-cell-count-and-typesdetection'
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
print(torch.__version__, torchvision.__version__)
os.getcwd()
%%time
# Here I have provided only 10 epochs, feel free to increment it, to get more accuracy !!!


!python yolov5/train.py --img 640 --batch 8 --epochs 100 --data yolov5/bcc-kaggle.yaml --cfg /kaggle/working/yolov5/models/yolov5s.yaml --name BCCM
# Detection or Run Inference on the model

!python yolov5/detect.py --source /kaggle/working/blood-cell-count-and-typesdetection/images/images/valid/ --weights ../input/blood-cell-count-and-typesdetection/best_BCCM.pt

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