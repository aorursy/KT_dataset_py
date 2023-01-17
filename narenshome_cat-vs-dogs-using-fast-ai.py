# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# !pip install "torch==1.4" "torchvision==0.5.0"
from fastai import *
from fastai.vision import *
np.random.seed(42)
data = ImageDataBunch.from_folder("/kaggle/input/dogs-cats-images/dataset/",
                                  train="training_set",
                                  test="test_set",
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(),
                                  size=224,
                                  bs=64).normalize(imagenet_stats)
#View some images
data.show_batch(rows=3, figsize=(5,5))
learner = cnn_learner(data, models.resnet34, metrics = error_rate, model_dir = '/kaggle/working/')
learner.fit_one_cycle(4)
print("accuracy = {}".format(100-0.011250))
learner.save('stage-1') #save model
interp = ClassificationInterpretation.from_learner(learner)
losses, idx = interp.top_losses()
len(data.valid_ds) == len(losses) == len(idx)
data.classes, data.c
#Misclassified Categories
interp.plot_top_losses(9, figsize=(10,10))
