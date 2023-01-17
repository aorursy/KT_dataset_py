%reload_ext autoreload

%autoreload 2

%matplotlib inline
from pathlib import Path

from fastai.vision import *

from fastai.metrics import error_rate

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
root_path = Path('../input/surat batak')

image_size = 224

batch_size = 16
data_batak = {str(i).split('/')[-1].split()[0]: i for i in root_path.ls()}
data_batak
db = {}

np.random.seed(1)

for name in data_batak:

    print("read", data_batak[name])

    db[name] = ImageDataBunch.from_folder(data_batak[name],  

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=10),

                                  size=image_size,bs=batch_size, 

                                  num_workers=0).normalize(imagenet_stats)
learners = {}

max_lr = {

    'Mandailing': (1e-7,1e-4),

    'Karo': (1e-7,1e-4),

    'Toba': (1e-7,1e-4),

    'Simalungun': (1e-7,1e-4),

    'Pakpak': (1e-8,1e-6)

}

for name in data_batak:

    print("train", data_batak[name])

    learn = cnn_learner(db[name], models.resnet18, metrics=accuracy, model_dir="/tmp/model/")

    learn.fit_one_cycle(10)

    learn.save('batak-{}-stage-1'.format(name))

    learn.unfreeze() 

    learn.fit_one_cycle(30, max_lr=slice(*max_lr[name]))

    learn.save('batak-{}-stage-2'.format(name))

    learners[name] = learn

    
metrics = {}

for name in data_batak:

    print("validation", name)

    metrics[name] = learners[name].validate()
print("Accuracy")

for name in data_batak:

    print("{:11}: {:.3f}".format(name, metrics[name][1]))
name = 'Pakpak'

db_current = db[name]

learn = learners[name]
learn.summary()
db_current.show_batch(rows=4, figsize=(10,10))
print(db_current.classes)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(db_current.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(10,10))
interp.most_confused(min_val=0.1)
interp.plot_confusion_matrix()