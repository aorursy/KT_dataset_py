# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
from fastai.vision import *
from fastai.metrics import error_rate
import warnings
warnings.filterwarnings("ignore")
bs = 64
path = Path('../input/face-mask-12k-images-dataset/Face Mask Dataset/'); path
data = ImageDataBunch.from_folder(path,train = 'Train',test= 'Test', valid = 'Validation',ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
data.show_batch(rows=4, figsize=(7,7))
data.label_list
learn = cnn_learner(data, models.resnet18, metrics=error_rate)
learn.fit(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(25,25), dpi=60)
interp.plot_top_losses(3,figsize=(15,11))