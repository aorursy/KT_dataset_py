# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))
        

# Any results you write to the current directory are saved as output.
from tqdm import tqdm_notebook as tqdm
from fastai import *
from fastai.vision import *
from pathlib import *
base_path=Path('/kaggle/input/plant-pathology-2020-fgvc7/')
def get_tag(row):
    if row.healthy:
        return "healthy"
    if row.multiple_diseases:
        return "multiple_diseases"
    if row.rust:
        return "rust"
    if row.scab:
        return "scab"
def transform_data(train_labels):
    train_labels.image_id = [image_id+'.jpg' for image_id in train_labels.image_id]
    train_labels['tag'] = [get_tag(train_labels.iloc[idx]) for idx in train_labels.index]
    train_labels.drop(columns=['healthy', 'multiple_diseases', 'rust', 'scab'], inplace=True)
train_labels = pd.read_csv(base_path/"train.csv")
path = base_path/"images"
train_labels.head()
transform_data(train_labels)
train_labels = train_labels.set_index("image_id")

train_labels.head()
src = (ImageList.from_folder(path)
      .filter_by_func(lambda fname: "Train" in fname.name) 
      .split_by_rand_pct(0.2)
      .label_from_func(lambda o: train_labels.loc[o.name]['tag']))
tfms = get_transforms(flip_vert=True, max_zoom=2.)
data_224 = (src
        .transform(tfms=tfms, size=224)
        .databunch(bs=20).normalize(imagenet_stats))
data_448 = (src
        .transform(tfms=tfms, size=448)
        .databunch(bs=12).normalize())
data_224.show_batch(6,figsize=(14,12))
learn = cnn_learner(data_224, models.resnet50, metrics=[accuracy], wd=1e-3,model_dir="/kaggle/working")
learn.path = Path('/')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(6,max_lr=slice(3e-3,3e-4),callbacks=[callbacks.EarlyStoppingCallback(learn,monitor='valid_loss', min_delta=0.1, patience=2)])
learn.fit_one_cycle(1,max_lr=slice(3e-3,3e-4),callbacks=[callbacks.EarlyStoppingCallback(learn,monitor='valid_loss', min_delta=0.1, patience=2)])
#learn.fit_one_cycle(6)
learn.fit_one_cycle(6)
learn.save("stage-1-224")
learn.fit_one_cycle(4,max_lr=slice(1e-05,3e-4))
learn.load("stage-3-448")
learn.save("stage-3-448")
learn.freeze()
learn.data=data_448
learn.fit_one_cycle(4)
preds, y = learn.get_preds(DatasetType.Test)
free = gpu_mem_get_free_no_cache()
learn.freeze()
test_images = ImageList.from_folder(base_path/"images")
test_images.filter_by_func(lambda x: x.name.startswith("Test"))
test_df = pd.read_csv(base_path/"test.csv")
test_df['healthy'] = [0.0 for _ in test_df.index]
test_df['multiple_diseases'] = [0.0 for _ in test_df.index]
test_df['rust'] = [0.0 for _ in test_df.index]
test_df['scab'] = [0.0 for _ in test_df.index]
test_df = test_df.set_index('image_id')
        
for item in tqdm(test_images.items):
        name = item.name[:-4]
        img = open_image(item)
        preds = learn.predict(img)[2]

        test_df.loc[name]['healthy'] = preds[0]
        test_df.loc[name]['multiple_diseases'] = preds[1]
        test_df.loc[name]['rust'] = preds[2]
        test_df.loc[name]['scab'] = preds[3]
            
test_df.to_csv(f"/kaggle/working/result.csv")



