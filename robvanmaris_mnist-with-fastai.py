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
# Import fastai libraries

from fastai.vision import *

from fastai.metrics import error_rate
!cd /kaggle/input

!pwd
# Uncompress the archives containing the images.

# The resulting datasets will be in:

#    /kaggle/input/trainingSet

#    /kaggle/input/testSet

# The trainingSet directory will contain subdirectories corresponding to the labels 0..9. 





!tar -zxf ../input/mnistasjpg/trainingSet.tar.gz

!tar -xxf ../input/mnistasjpg/testSet.tar.gz 

!mv trainingSet ../input

!mv testSet ../input

!ls -li /kaggle/input



# Note: this could have been done as well using fastai utility method untar_data().
batchsize = 64

path_training = Path('/kaggle/input/trainingSet')

path_test = Path('/kaggle/input/testSet')
data = ImageDataBunch.from_folder(path_training, valid_pct=0.15, seed=123, 

                                  test=path_test, size=28,

                                  bs=batchsize, ds_tfms=get_transforms(do_flip=False))

data.normalize(mnist_stats)

data.show_batch(rows=3, figsize=(7, 6))

print('Available labels: {}'.format(data.classes))
model = models.resnet34 # Results in error_rate 0.0046 with 4 epochs plus full retraining for 12 epochs with max_lr=slice(1e-4, 1e-3)

# model = models.resnet50 # 0.052 with 4 epochs plus full retraining for 12 epochs with max_lr=slice(1e-4, 1e-3)
learn = cnn_learner(data, model, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1-resnet34-4-epochs')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()
learn.load('stage-1-resnet34-4-epochs')

learn.unfreeze()
learn.fit_one_cycle(12, max_lr=slice(1e-4, 1e-3))
learn.save('resnet34-4-12-1e-4-1e-3')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(36, figsize=(12,12))
interp.plot_confusion_matrix()
interp.most_confused(min_val=2)
test_results = pd.DataFrame(columns=['ImageId','Label'])

for i in range(1, 28000 + 1):

    img = open_image(path_test/'img_{}.jpg'.format(i))

    prediction = learn.predict(img)[1].item()

    test_results.loc[i] = [i, prediction]
test_results.head()
test_results.to_csv('submission.csv', index=False)