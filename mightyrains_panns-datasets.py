# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git
!pip download torchlibrosa
# !wget https://zenodo.org/record/3960586/files/Cnn10_mAP%3D0.380.pth 

!wget https://zenodo.org/record/3960586/files/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth 

# !wget https://zenodo.org/record/3960586/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth 

!wget https://zenodo.org/record/3960586/files/CNN14_emb128_mAP%3D0.412.pth 

# !wget https://zenodo.org/record/3960586/files/Cnn14_emb32_mAP%3D0.364.pth 

!wget https://zenodo.org/record/3960586/files/Cnn14_emb512_mAP%3D0.420.pth 

!wget https://zenodo.org/record/3960586/files/Cnn14_mAP%3D0.431.pth 

# !wget https://zenodo.org/record/3960586/files/Cnn6_mAP%3D0.343.pth 

# !wget https://zenodo.org/record/3960586/files/DaiNet19_mAP%3D0.295.pth 

#!wget https://zenodo.org/record/3960586/files/LeeNet11_mAP%3D0.266.pth

# !wget https://zenodo.org/record/3960586/files/LeeNet24_mAP%3D0.336.pth 

# !wget https://zenodo.org/record/3960586/files/MobileNetV1_mAP%3D0.389.pth 

# !wget https://zenodo.org/record/3960586/files/MobileNetV2_mAP%3D0.383.pth 

# !wget https://zenodo.org/record/3960586/files/Res1dNet31_mAP%3D0.365.pth 

# !wget https://zenodo.org/record/3960586/files/Res1dNet51_mAP%3D0.355.pth 

!wget https://zenodo.org/record/3960586/files/ResNet22_mAP%3D0.430.pth 

!wget https://zenodo.org/record/3960586/files/ResNet38_mAP%3D0.434.pth 

!wget https://zenodo.org/record/3960586/files/ResNet54_mAP%3D0.429.pth 

# !wget https://zenodo.org/record/3960586/files/Wavegram_Cnn14_mAP%3D0.389.pth 

!wget https://zenodo.org/record/3960586/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth 
!ls