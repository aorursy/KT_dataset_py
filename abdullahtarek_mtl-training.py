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
!git clone https://github.com/ahmadelsallab/MultiCheXNet.git
from MultiCheXNet.data_loader.MTL_dataloader import get_train_validation_generator
det_csv_path = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
seg_csv_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
det_images_path = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/"
seg_images_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"
train_gen,val_gen = get_train_validation_generator(det_csv_path,seg_csv_path , det_images_path, seg_images_path)

from MultiCheXNet.utils.ModelBlock import ModelBlock
from MultiCheXNet.utils.Encoder import Encoder
from MultiCheXNet.utils.Detector import Detector
from MultiCheXNet.utils.Segmenter import Segmenter
from MultiCheXNet.utils.Classifier import Classifier

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from MultiCheXNet.utils.loss.MTL_loss import MTL_loss
from tensorflow.keras.optimizers import Adam
encoder = Encoder( ) 
classifier = Classifier(encoder)
img_size = 256
n_classes=1
detector=Detector(encoder,img_size, n_classes)
segmenter = Segmenter(encoder)
MTL_model = ModelBlock.add_heads(encoder,[classifier,detector,segmenter ])
#from tensorflow.keras.utils import plot_model
#plot_model(MTL_model)
#MTL_model.summary()


classification_loss= "categorical_crossentropy"
detection_loss= detector.loss
segmentation_loss= segmenter.loss

mtl_loss= MTL_loss(classification_loss , detection_loss ,segmentation_loss)
lossWeights = [1.0,1.0,1.0]
INIT_LR = 1e-4
EPOCHS =20

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

MTL_model.compile(optimizer=opt, loss=mtl_loss, metrics=[])
MTL_model.fit_generator(train_gen)



