import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(filename))
!pip install /kaggle/input/keras-pretrained-imagenet-weights/image_classifiers-1.0.0-py3-none-any.whl
# for keras

from classification_models.tfkeras import Classifiers



root = '/kaggle/input/keras-pretrained-imagenet-weights/'

Classifiers.models_names()
ResNet18, preprocess_input = Classifiers.get('resnet18')

model = ResNet18(include_top = False, input_shape=(224, 224, 3), weights=root + 'resnet18_imagenet_1000_no_top.h5')
SeResNeXT, preprocess_input = Classifiers.get('seresnext50')

model = SeResNeXT(include_top = False, input_shape=(224, 224, 3), weights=root + 'seresnext50_imagenet_1000_no_top.h5')