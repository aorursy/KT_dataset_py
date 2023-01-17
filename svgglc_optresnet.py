# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import fastai

from fastai.vision import *

from fastai.callbacks.hooks import *
#path = Path('../input/data/data')

path = Path('../input/dataset2/data')
train = path/'train'

test =  path/'test'
path.ls()
#doc(get_transforms)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train="train",valid_pct=0.2,

ds_tfms=get_transforms(do_flip=False,flip_vert=True,max_rotate=0, max_zoom=0.3,

                       max_lighting=0, max_warp=0), size=[150,400],bs=128, num_workers=4).normalize(imagenet_stats)
data.classes
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(5)
learn = cnn_learner(data, models.resnet50, metrics=accuracy,pretrained=True)
learn.fit_one_cycle(15)
learn.model_dir ='/kaggle/working'

learn.export('/kaggle/working/r12.pkl')

learn.save('r4')
learn.freeze()

learn.unfreeze()

learn.lr_find();learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(14)

learn.unfreeze()

learn.export('/kaggle/working/r12_2.pkl')

learn.save('r1_3')
learn.lr_find();learn.recorder.plot(suggestion=True)
learn.recorder.plot_losses()
learn.unfreeze()



learn.lr_find();learn.recorder.plot(suggestion=True)



learn.export('/kaggle/working/r12_3.pkl')

learn.save('r1_2')
learn.lr_find();learn.recorder.plot()
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
#doc(interp.plot_top_losses)

interp.plot_confusion_matrix(figsize=(8,8))
interp.most_confused(min_val=2)