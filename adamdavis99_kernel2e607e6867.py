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
path=Path('/kaggle/input/cat-and-dog')
path.ls()
data=ImageList.from_folder(path)
data
data=data.split_by_folder(train='training_set',valid='test_set')
data
data=data.label_from_folder()

data
data=data.transform(tfms=get_transforms(),size=224)
d1=data.databunch(bs=64)
d1
d1.show_batch(4,figsize=(8,8))
learn = cnn_learner(d1, models.densenet161, metrics=[error_rate,accuracy], model_dir='/tmp/models').to_fp16()
learn.fit_one_cycle(4)
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.unfreeze()
learn.fit_one_cycle(5,max_lr=slice(6e-7,6e-6))
interp=ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.save('model1')
learn.export(Path("/kaggle/working/export.pkl"))
cat,pred,_=learn.predict(d1.valid_ds.x[0])
d1.classes[pred]
d1.classes
show_image(d1.valid_ds.x[0])

d1.valid_ds.y[0]
