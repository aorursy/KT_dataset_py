import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

from fastai import *

from fastai.vision import *

import os
print(os.listdir("../input"))
path = Path("../input")
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
np.random.seed(42)

src = (ImageList.from_csv(path, 'train.csv', folder='train/train')

      .split_by_rand_pct(0.2)

      .label_from_df())
data = (src.transform(tfms, size=128)

       .databunch()

       .normalize(imagenet_stats))
#data.show_batch(rows=3, fig_size=(7,6))
arch = models.resnet50
learn = cnn_learner(data, arch, metrics=accuracy, model_dir="/tmp/model/")
#learn.lr_find()

#learn.recorder.plot()
lr = 3e-04

learn.fit_one_cycle(7, slice(lr))
learn.unfreeze()
#learn.lr_find()

#learn.recorder.plot()
learn.fit_one_cycle(7, slice(1e-5, lr/5))
test = ImageList.from_csv(path, 'sample_submission.csv', folder='test/test')
data.add_test(test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
test_df = pd.read_csv("../input/sample_submission.csv")
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)