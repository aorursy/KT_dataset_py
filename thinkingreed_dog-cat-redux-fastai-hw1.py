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

        break

    break



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
DATA_PATH = "../input/dogs-vs-cats-redux-kernels-edition"

TRAIN_PATH = DATA_PATH + "/train"

TEST_PATH = DATA_PATH + "/test"

path = untar_data("",DATA_PATH,DATA_PATH); path; path.ls()
fnames = get_image_files(TRAIN_PATH)

pat = r'([^/]+)\.\d+\.jpg$'

fnames[:5]
bs=64

data = ImageDataBunch.from_name_re(TRAIN_PATH, fnames, pat, ds_tfms=get_transforms(), size=499, bs=bs

                                  ).normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet18, metrics=error_rate, pretrained=True)
learn.model_dir = "/tmp"

os.access('/tmp', os.W_OK)
learn.fit_one_cycle(3)

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)
learn.unfreeze()

learn.fit_one_cycle(1)

learn.load('stage-1')

learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-5))
learn.save('stage-tunned-34')
test_fnames = get_image_files(TEST_PATH)

test_fnames[:5]
#for f in test_fnames:

img = open_image(test_fnames[3])

learn.predict(img)
img
learn.predict(img)[2][1].item()
submission = {"id":[],"label":[]}

idpat = r''

for f in test_fnames:

    #print(f)

    img = open_image(f)

    fname = f.name

    id = fname[:fname.rfind(".")]

    #print(id)

    #break

    submission["id"].append(id)

    submission["label"].append(learn.predict(img)[2][1].item())
result = pd.DataFrame(submission)

result.to_csv('submission.csv', index=False)