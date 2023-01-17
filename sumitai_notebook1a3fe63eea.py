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
!pip uninstall keras -y

!pip install git+https://github.com/qubvel/segmentation_models

!git clone https://github.com/SlinkoIgor/ImageDataAugmentor.git
from segmentation_models import Unet

import segmentation_models as sm



from tensorflow.keras.layers import Input, Conv2D

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

np.random.seed(0)



base_model = Unet(backbone_name='efficientnetb0',

                  encoder_weights='imagenet',

                  classes=4, 

                  activation='softmax')
base_model.summary()
import segmentation_models as sm
from segmentation_models import Unet

model = Unet('resnet34')
model.summary()