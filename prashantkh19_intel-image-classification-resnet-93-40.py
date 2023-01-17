from fastai import *

from fastai.vision import *

import PIL
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

names = []

labels = []

broken_images = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        folder = (dirname.split('/')[5]).split('_')[1]

        if(folder == 'pred'):

            continue

        label = dirname.split('/')[6]

        try:

            img = PIL.Image.open(os.path.join(dirname, filename)) 

            if (not (img.size == (150, 150))):

                continue

            names.append(os.path.join(dirname[14:], filename))

            labels.append(label)

        except (IOError, SyntaxError) as e:

            print('Bad file:', os.path.join(dirname, filename))

            broken_images.append(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
df = pd.DataFrame({

    "names": names,

    "labels": labels})

df.head()
data = ImageDataBunch.from_df('/kaggle/input', df)

data.normalize()
data.show_batch(rows=3, figsize=(5,5))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)