!git clone https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection.git
import os
os.chdir('ssd_tensorflow_traffic_sign_detection')
!pip install tensorflow==1.13.1
import tensorflow as tf
print(tf.__version__)
!pip install moviepy
!python inference.py -m demo
