# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# print(os.listdir("../input/aerial-cactus-identification"))





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision import *

from fastai.metrics import error_rate

bs = 64

import os

print(os.listdir("../input"))
# ! rmdir -r /kaggle/working/train

! unzip -q /kaggle/input/aerial-cactus-identification/train.zip

! unzip -q /kaggle/input/aerial-cactus-identification/test.zip

! cp /kaggle/input/aerial-cactus-identification/train.csv /kaggle/working/

! cp /kaggle/input/aerial-cactus-identification/sample_submission.csv /kaggle/working/
batch_size=64

data_path = '/kaggle/working/'

data_path_train = data_path + 'train/'

data_path_test = data_path + 'test/'

df_train = pd.read_csv(data_path + "train.csv")

df_test = pd.read_csv(data_path + "sample_submission.csv")
df_train.head()
data_path_train
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)



# I have built this using the Data-Block API - so convenient!

data=(ImageList.from_df(path=data_path_train, df=df_train)

      .split_by_rand_pct(valid_pct = 0.2, seed = 42)

      .label_from_df(cols = 'has_cactus')

      .transform(tfms, size=128)

      .databunch()    

     )

data.add_test(ImageList.from_df(df_test, path=data_path_test))



data
data.show_batch(rows = 3, figsize = (5,5))

learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/").to_fp16()
learn.model
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, max_lr=slice(3e-03, 3e-2))

learn.save('stage-1-resnet34-128-bs64')

# I have built this using the Data-Block API - so convenient!

# It is probably the easiest way I have seen so far to build an input pipeline, in my journey.

data=(ImageList.from_df(path=data_path_train, df=df_train)

      .split_by_rand_pct(valid_pct = 0.2, seed = 42)

      .label_from_df(cols = 'has_cactus')

      .transform(tfms, size=256)

      .databunch()    

     )

data.add_test(ImageList.from_df(df_test, path=data_path_test))

learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/").to_fp16()



learn.freeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, max_lr=slice(1e-03, 1e-2))
learn.save('stage-1-resnet34-256-bs64')



learn.freeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, max_lr=slice(1e-06, 5e-5))
learn.save('stage-2-resnet34-256-bs64')



predictions = learn.get_preds(ds_type=DatasetType.Test)[0]
! rm -r /kaggle/working/train

! rm -r /kaggle/working/test
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes)

type(predicted_classes)
# reload sample_submission

sample_submission=pd.read_csv("sample_submission.csv")

sample_submission.head(10)

# output = pd.DataFrame({sample_submission.columns[0]: [id for id in df_test.id],

#                       sample_submission.columns[1]: predicted_classes.numpy()})

output=df_test.copy()

output["has_cactus"]=predicted_classes.numpy()

output.to_csv('submission.csv', index=False)

output.to_csv('/kaggle/working/submission.csv', index=False)

output.head(10)