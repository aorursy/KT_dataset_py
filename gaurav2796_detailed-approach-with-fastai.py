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
# importing our dependencies / Packages

from fastai import *

from fastai.vision import *

import pandas as pd

import numpy as np
# viewing our data

data_folder = Path("../input")

data_folder.ls()
# Getting the data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
test_img = ImageList.from_df(test_df, # test data frame

                             path=data_folder/'test', 

                             folder='test' 

                            )



train_img = (

    ImageList.from_df(train_df, #train data frame

                      path=data_folder/'train', 

                      folder='train')

        .split_by_rand_pct(0.2) # making 20% of validation dataset

        .label_from_df() # it labels according to dataframe passed to it

        .add_test(test_img) # adding test data

        .transform(get_transforms(flip_vert=True), size=128) # adding transforms

        .databunch(path='.', bs=64) # create databunch // path is used internally to store temporary files // bs = batch size

        .normalize(imagenet_stats) #  normalise according to pretained model

       )
learn = cnn_learner(train_img, #training data

                    models.resnet34,#model

                    metrics=[error_rate, accuracy] #error rate

                   )
learn.fit_one_cycle(4) #learning for 4 epochs
learn.save('model-1') # saving the model
interp = ClassificationInterpretation.from_learner(learn)
# for plotting top losses

losses,idxs = interp.top_losses() 

interp.plot_top_losses(9,figsize=(15,11))
# to view confusion matrix of model

interp.plot_confusion_matrix()
learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-05))
learn.validate()
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission.csv', index=False)