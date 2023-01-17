import os

!pip install split_folders

import split_folders
data_dir = "../input/cgiar-computer-vision-for-crop-disease/ICLR/train/train/"
split_folders.ratio(input=data_dir, output="train_val", seed=42, ratio=(0.8, 0.2) )