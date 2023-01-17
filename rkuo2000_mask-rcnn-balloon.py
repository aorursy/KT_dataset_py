#!git clone https://github.com/matterport/Mask_RCNN # TF1.x
!git clone https://github.com/akTwelve/Mask_RCNN    # TF2.x
%cd Mask_RCNN
#!pip install -q tensorflow==1.15.0
#!pip install -q keras==2.0.8
!pip install -q pycocotools
## download pre-trained COCO weights
#!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
## Download Balloon dataset
!mkdir datasets
%cd datasets
!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
!unzip balloon_dataset.zip
!rm balloon_dataset.zip
%cd ..
%cd samples/balloon
#!python balloon.py train --dataset ../../datasets/balloon --weights=coco
## download pre-trained Balloon weights
!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
!mv mask_rcnn_balloon.h5 ../..
from IPython.display import Image
# pick an image from balloon/val
Image('../../datasets/balloon/val/3825919971_93fb1ec581_b.jpg')
!python balloon.py splash --dataset ../../datasets/balloon --weights ../../mask_rcnn_balloon.h5 --image '../../datasets/balloon/val/3825919971_93fb1ec581_b.jpg'
Image('splash_20201005T135605.png')