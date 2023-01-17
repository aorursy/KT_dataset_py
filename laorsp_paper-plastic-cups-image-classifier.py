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
from fastai.vision import *
path = Path('data/cups')

dest = path/'plastic'

dest.mkdir(parents=True, exist_ok=True)
download_images(Path('../input/urls_plastic.txt'), dest, max_pics = 400)
dest = path/'paper'

dest.mkdir(parents=True, exist_ok=True)

download_images(Path('../input/urls_paper.txt'), dest, max_pics = 400)
verify_images(path/'plastic', delete=True, max_size=500)

verify_images(path/'paper', delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(6,7))
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5,max_lr=2e-4)
learn.save('stage-2')
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(10)
from fastai.widgets import *
ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
learn.fit_one_cycle(5,max_lr=(2e-4)/5)
learn.save('stage-3')
learn.unfreeze()
learn.fit_one_cycle(10,max_lr=slice(2e-4,(2e-4)/5))
learn.save('final')
learn.load('final')