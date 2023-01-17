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
from fastai import *
from fastai.vision import *
path = Path('../input/intel-image-classification/seg_train/seg_train')
import torch
defaults.device = torch.device('cuda')
path.ls()
data = ImageDataBunch.from_folder(path,valid_pct=0.2,ds_tfms=get_transforms(),size=224,num_workers=2).normalize(imagenet_stats)
# just for practrice
# data1 = (ImageList.from_folder(path)
#         .split_by_rand_pct(.2)
#         .label_from_folder()
#         .transform(get_transforms(),size=224)
#         .databunch(bs=20)
#         .normalize(imagenet_stats))
data.show_batch()
classes =['glacier','sea','forest','street','mountain','buildings']  ## deifining labels of the images

for c in classes:
    print(c)
    verify_images(path/c,delete=True,max_size=500)
learn = cnn_learner(data,models.resnet34,metrics=[error_rate])
learn.fit_one_cycle(5)
learn.model_dir='/kaggle/working'
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2,1e-5)
learn.save('stage-2')
#Niceeeee
learn.fit_one_cycle(2,1e-5)
interpret = ClassificationInterpretation.from_learner(learn)
interpret.plot_confusion_matrix()
interpret.plot_top_losses(9,figsize=(15,11))
learn.show_results()
interpret.most_confused(min_val=5) 
import os
i=0
preds=[]
pred_path = '../input/testing-data'
for f in os.listdir(pred_path):
    i+=1
    if i<10:
        print(f)
        img = open_image(os.path.join(pred_path,f))
        pred = learn.predict(img)
        preds.append((img,pred))
data.classes
learn.export(Path('/kaggle/working/new.pkl'))
preds