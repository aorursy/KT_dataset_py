!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -r yolov5/requirements.txt  # install dependencies
%cd yolov5

import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
%cd ..
# Download wheat and labels
gdrive_download('1XqetalfvoOpgYK2YsyHmMu85qEHTgjGN','WheatYolo.zip')  # dataset
!mv ../input/wheatyaml/yolov5xnc1.yaml ./
%cd yolov5
!ls
%%time
!python train.py --img 416 --batch 16 --epochs 2 --data '../data.yaml' --cfg '../yolov5xnc1.yaml' --weights '' --name tutorial --nosave --cache
# start tensorboard
# launch after training 
# log saves in the folder "runs"
#%load_ext tensorboard
#%tensorboard --logdir runs
#if Kaggle ever fixes issue with tensorboard uncomment the two lines above. 
Image(filename='runs/exp0_tutorial/train_batch1.jpg', width=900)  # view augmented training mosaics
Image(filename='runs/exp0_tutorial/test_batch0_gt.jpg', width=900)  # view test image labels
Image(filename='runs/exp0_tutorial/test_batch0_pred.jpg', width=900)  # view test image predictions
# trained weights are saved by default in our weights folder
%ls
!python detect.py --source ../test/images --weights 'runs/exp0_tutorial/weights/last_tutorial.pt' --img 416 --conf 0.5
!ls
%cd inference/output/
!ls
Image(filename='2fd875eaa_jpg.rf.dd1d8cd790ac7bd4fa42b63bd6b6293b.jpg', width=900)
Image(filename='348a992bb_jpg.rf.1d04a696a55ee0ae7737b687d63f7af6.jpg', width=900)