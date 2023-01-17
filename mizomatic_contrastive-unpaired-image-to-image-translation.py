# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob

import shutil

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

cwd = os.getcwd()

from random import sample

from PIL import Image

from IPython.display import HTML
HTML('<iframe width="840" height="560" src="https://www.youtube.com/embed/jSGOzjmN8q0?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
!git clone https://github.com/sumanyumuku98/contrastive-unpaired-translation.git CUT

# !pip install -r requirements.txt

# os.chdir(cwd)
os.chdir("CUT/")

!pip install -r requirements.txt

os.chdir(cwd)
os.chdir("/kaggle/working/CUT/datasets/")

os.mkdir("images")

os.chdir("images")

os.mkdir("trainA")

os.mkdir("trainB")

# os.chdir(cwd)
trainA = glob.glob("/kaggle/input/gan-getting-started/photo_jpg/*")

trainB = glob.glob("/kaggle/input/gan-getting-started/monet_jpg/*")
# os.chdir("/kaggle/working/CUT/datasets/images/")

# !ls
# Sample Few images from photos_jpg as required. Here I have sampled 300 out of 7038 images.

trainA_sample = sample(trainA,300)

trainB_sample = trainB

for file in trainA_sample:

    imgName = file.split("/")[-1]

    newPath = os.path.join("./trainA/",imgName)

    shutil.copyfile(file,newPath)

for file in trainB_sample:

    imgName = file.split("/")[-1]

    newPath = os.path.join("./trainB/",imgName)

    shutil.copyfile(file,newPath)
os.chdir(cwd)

os.chdir("/kaggle/working/CUT/")
# Change the no. of epochs and decay accordingly

!python train.py --dataroot ./datasets/images --name monet_CUT --CUT_mode FastCUT --n_epochs 1 --n_epochs_decay 1 --save_epoch_freq 1
os.mkdir("./checkpoints/monet_test")
!cp ./checkpoints/monet_CUT/latest_net_G.pth ./checkpoints/monet_test/
!ln -s /kaggle/input/gan-getting-started/photo_jpg ./datasets/images/testA
!ln -s /kaggle/input/gan-getting-started/monet_jpg ./datasets/images/testB
!python test.py --dataroot /kaggle/working/CUT/datasets/images --name monet_test --CUT_mode FastCUT --num_test 7037
# lister = glob.glob("./results/monet_test/test_latest/images/fake_B/*")
shutil.make_archive("/kaggle/working/images","zip","./results/monet_test/test_latest/images/fake_B/")
!du -sh /kaggle/working/images.zip
os.chdir(cwd)
!rm -r CUT
# !ls