!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -qr yolov5/requirements.txt  # install dependencies (ignore errors)
%cd yolov5

import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
# Download COCO val2017
gdrive_download('1Y6Kou6kEB0ZEMCCpJSKStCor4KAReE43','coco2017val.zip')  # val2017 dataset
!mv ./coco ../  # move folder alongside /yolov5
# Run YOLOv5x on COCO val2017
!python test.py --weights runs/exp1/weights/best.pt --data kkctbn.yaml --img 416
!mkdir ../kkctbn
!cp -r /kaggle/input/kkctbn ../kkctbn
!mkdir ../kkctbn/images
!mkdir ../kkctbn/labels
!mv ../kkctbn/dataset/*.txt ../kkctbn/labels
!mv ../kkctbn/dataset/*.jpg ../kkctbn/images
!cp /kaggle/input/kkctbn/kkctbn.yaml ./data
!cp /kaggle/input/kkctbn/yolov5s.yaml ./models/yolov5s2.yml
!ls ./models
!ls data
# Start tensorboard (optional)
%load_ext tensorboard
%tensorboard --logdir runs
# Train YOLOv5s on coco128 for 3 epochs
!python train.py --img 416 --batch 16 --epochs 50 --data kkctbn.yaml --cfg yolov5s2.yml --weights yolov5s.pt --nosave --cache
Image(filename='runs/exp0/train_batch1.jpg', width=900)  # view augmented training mosaics
Image(filename='runs/exp0/test_batch0_gt.jpg', width=900)  # view test image labels
!ls runs/exp1
Image(filename='runs/exp1/results.png', width=900)
Image(filename='runs/exp1/test_batch0_pred.jpg', width=900)  # view test image predictions
from utils.utils import plot_results; plot_results()  # plot results.txt files as results.png