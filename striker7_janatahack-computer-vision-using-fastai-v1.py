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
# Recently FastAI is updated to version v2 so this code will update you on what has been changed
# remember in kaggle you have to turn internet to download the new packages (settings->internet(on))
import pandas as pd
import numpy as np
import fastai; print(fastai.__version__)
from fastai.vision import *
path = '/kaggle/input/train-and-test-files/train'
train_data = pd.read_csv('/kaggle/input/train-and-test-files/train/train.csv')
test_data = pd.read_csv('/kaggle/input/train-and-test-files/test.csv')
# this will add flip, warp, zoom and rotation to the images
tfms = get_transforms(do_flip = True, max_lighting = 0.2, max_zoom= 1.1, max_warp = 0.15, max_rotate = 45)
emergency = ImageDataBunch.from_csv(path, folder= 'images', 
                              csv_labels = 'train.csv',
                              ds_tfms = tfms,
                              fn_col = 'image_names',
                              label_col = 'emergency_or_not',
                              bs = 16,
                              size = 300).normalize(imagenet_stats)
emergency.show_batch()
# 0 indicating not emergency
# 1 indicating emergency
# we will be using resnet 101 model
learn = cnn_learner(emergency, models.resnet101, metrics=[accuracy])
import torch
torch.cuda.set_device(0)
#fir the data in the model
# unfreeze will unfreeze all your layers
# freeze will stop training of all the layers except last one
learn.unfreeze()
learn.fit_one_cycle(3)
learn.freeze()
learn.fit_one_cycle(6)
import os
os.makedirs('./test')
import shutil


def save_file_folder_test(x):
  o = path + '/images/'+x['image_names']
  shutil.copy(o, './test')

test_data.apply(save_file_folder_test, axis=1)
#save this 
fnames = get_image_files('./test')
dl = ImageList.from_folder('./test')
final=[learn.predict(i)[1].item() for i in dl]
final_submit = pd.DataFrame(columns=['image_names', 'emergency_or_not'])
final_submit['image_names'] =[str(i).split('/')[1] for i in fnames]
final_submit['emergency_or_not'] = final
final_submit.to_csv('./submit.csv', index=False)