#!git clone https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch

!git clone https://github.com/rkuo2000/Yolov5_DeepSort_Pytorch

%cd Yolov5_DeepSort_Pytorch
# install requirements

!pip install easydict

!pip install tdqm
!cp -rf ../../input/yolov5-weights-r30/* yolov5/weights
!cp ../../input/deepsort-ckpt/ckpt.t7 deep_sort/deep_sort/deep/checkpoint
!ls deep_sort/deep_sort/deep/checkpoint
!python track.py --source /kaggle/input/input-video/traffic720p-india.mp4 --img-size 640 --out ../output --save-txt
!ls -l ../output/results.mp4
import pandas as pd



df=pd.read_csv("../output/results.txt",delimiter=" ")

df.head()