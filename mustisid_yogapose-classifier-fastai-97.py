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
import warnings
warnings.filterwarnings('ignore')
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from fastai.vision.all import *
path = Path('../input/yoga-classes/DATASET')
data = ImageDataLoaders.from_folder(path, train='TRAIN', valid='TEST', valid_pct=None, seed=None, vocab=None, item_tfms=Resize(400), batch_tfms=aug_transforms(size=224, min_scale=0.75), bs=20, val_bs=None, shuffle_train=True, device=None)
data.show_batch()
learn = cnn_learner(data,resnet34,metrics=[accuracy])
learn.model_dir='/kaggle/output'
learn.lr_find()
learn.fine_tune(4,base_lr=1e-3,freeze_epochs=2)
intrep = ClassificationInterpretation.from_learner(learn)
intrep.most_confused(min_val=5)
intrep.plot_top_losses(5, nrows=1)
learn.save('stage1-96%-1e-3')
learn.load('stage1-96%-1e-3')
learn.fine_tune(2,1e-6)
learn.save('stage2-97%')
learn.model_dir='/kaggle/output'