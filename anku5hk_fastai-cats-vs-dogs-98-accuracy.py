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

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import accuracy

import numpy as np
np.random.seed(42)

training = '/kaggle/input/dogs-cats-images/dataset/training_set/'

path = '/kaggle/input/dogs-cats-images/dataset/'
data = ImageDataBunch.from_folder(Path(path), train='training_set', valid='test_set', ds_tfms = get_transforms(),

                                      size=224, bs=32)
data.show_batch(rows=3, figsize=(7,5))
learner = cnn_learner(data, models.resnet34, metrics = accuracy)
learner.fit_one_cycle(4)
learner.show_results()
y_preds, y, losses = learner.get_preds(with_loss = True)
interp = ClassificationInterpretation(learner, y_preds, y, losses)

interp.top_losses(5)
interp.plot_top_losses(5)
interp.plot_confusion_matrix()