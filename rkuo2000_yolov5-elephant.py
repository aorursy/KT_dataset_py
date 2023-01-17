!git clone https://github.com/ultralytics/yolov5
!mv yolov5/* ./
!python -m pip install --upgrade pip
!pip install -r requirements.txt
# Download YoloV5 pretrained models
#!weights/download_weights.sh
!cp -rf ../input/elephant-data/data/images data
!cp -rf ../input/elephant-data/data/labels data
!ls -l data/images
!ls -l data/labels
# Train Model
!python train.py --img 640 --batch 4 --epochs 30 --data ../input/elephant-data/data/elephant.yaml --cfg models/yolov5s.yaml --weight ""
# copy saved model to weights folder
!cp runs/exp0/weights/best.pt weights
# Detect Test Images
!python detect.py --source 'data/images/test' --weight weights/best.pt --output 'inference/output' 
import os
files = os.listdir('inference/output')
print(files)
from IPython.display import Image, clear_output  # to display images

Image(filename='inference/output/'+files[0], width=600)
Image(filename='inference/output/'+files[4], width=600)