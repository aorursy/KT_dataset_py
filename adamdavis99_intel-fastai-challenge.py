# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from fastai import *
from fastai.vision import *
path=Path('/kaggle/input/intel-image-classification')
np.random.seed(42)
tfms=get_transforms()
data=ImageDataBunch.from_folder(path,train='seg_train',valid='seg_test',size=64,num_workers=4,ds_tfms=tfms).normalize(imagenet_stats)
print(len(data.valid_ds))
print(data.classes)
print(data.c)
print(len(data.train_ds))
learn=cnn_learner(data,models.densenet161,metrics=error_rate)
learn.fit_one_cycle(5)
learn.unfreeze()
learn.model_dir = "/kaggle/working" 
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4,max_lr=slice(7.59e-8,7.59e-7))
learn.save('model-1')
interp=ClassificationInterpretation.from_learner(learn)
interp.top_losses()
interp.most_confused(min_val=5)
interp.plot_confusion_matrix()
learn.export(file = Path("/kaggle/working/export.pkl"))
imgs=['10017.jpg','10040.jpg','10059.jpg','10066.jpg','10052.jpg','10012.jpg','10043.jpg','10060.jpg','10038.jpg','10021.jpg','1003.jpg','10045.jpg','10004.jpg','10013.jpg','10069.jpg','10047.jpg','10034.jpg','10054.jpg','10048.jpg','10005.jpg']
ls=[]
for i in range(len(imgs)):
    pathimg=path/'seg_pred/seg_pred/{}'.format(imgs[i])
    image=open_image(pathimg)
    pred_class,pred_idx,outputs = learn.predict(image)
    print(pred_class)
    image.show()
    
pathimg=path/'seg_pred/seg_pred/{}'.format(imgs[11])
image=open_image(pathimg)

pred_class,pred_idx,outputs = learn.predict(image)
print(pred_class)
image
