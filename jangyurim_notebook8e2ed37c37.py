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
img_dir=os.listdir("../input/skin-cancer-mnist-ham10000")

img_dir
img_dir=os.listdir("../input/skin-cancer-mnist-ham10000/ham10000_images_part_1")

img_dir
img_dir=os.listdir("../input/skin-cancer-mnist-ham10000")

img_dir
df_train = pd.read_csv("../input/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv")
df_train
df_train.shape
df_train1 = pd.read_csv("../input/skin-cancer-mnist-ham10000/hmnist_28_28_L.csv")

df_train1
df_train2 = pd.read_csv("../input/skin-cancer-mnist-ham10000/hmnist_8_8_L.csv")

df_train2.shape
df_train2
df=pd.read_csv(("../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"))
df