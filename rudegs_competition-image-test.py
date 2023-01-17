# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

from IPython.display import Image



bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
# ls /kaggle/input/train_images/
# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



sample_submission_df = pd.read_csv("../input/sample_submission.csv")
train_df.head()
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_csv(path="../input/train_images/", csv_labels="../train.csv", fn_col="file_name", label_col="category_id", ds_tfms=tfms, size=32)
data.show_batch(rows=3)

# data.classes
# train_df["height"].value_counts()
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
help(cnn_learner)
learn.fit_one_cycle(1)
learn.model_dir='/kaggle/working/'

learn.save('stage-1')
learn.unfreeze()
# learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find()
# learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
img = test_df['file_name'][23678]

imgpath = '../input/test_images/' + img

# from IPython.display import Image

# Image(filename=imgpath)
# data_test = ImageDataBunch.from_csv(path="../input/test_images/", csv_labels="../test.csv", fn_col="file_name", ds_tfms=tfms, size=32)

# test_df
learn.predict(open_image(fn=imgpath))
data_test = ImageDataBunch.from_df(path="../input/test_images/", df=test_df, fn_col="file_name", ds_tfms=tfms, size=32)

# learn.data.add_test(data_test)

# test_df.head()
# learn.data.add_test(data_test) 

data_test
img_paths = test_df['file_name'].tolist()
preds = []



for i in range(0, len(img_paths)):

    file_name = img_paths[i];

    tens = learn.predict(open_image(fn='../input/test_images/'+ file_name))

    preds.append((file_name[:-4], int(tens[0]) ))

result = pd.DataFrame(preds, columns=['Id', 'Predicted'])
result.to_csv('submission.csv',index=False)