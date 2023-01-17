import os 
import numpy as np 
import pandas as pd 
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.vision.models import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

import torch 

data_root = '../input/airbus-ship-detection/'
path_train = os.path.join(data_root,'train_v2')
path_test = os.path.join(data_root,'test_v2')
# Get dataframe with label
masks = pd.read_csv(os.path.join(data_root, 'train_ship_segmentations_v2.csv'))
masks = masks[~masks['ImageId'].isin(['6384c3e78.jpg'])]  # remove corrupted image

df_clf = masks.groupby('ImageId').size().reset_index(name='counts')
df_clf = pd.merge(masks, df_clf)
df_clf['label'] = df_clf.apply(lambda c_row: 1 if isinstance(c_row['EncodedPixels'], str) else 0, 1)
df_clf = df_clf.drop(columns=['EncodedPixels','counts'])

# Prepare data
def get_data(bs=64, size=256, split=0.2):
   
    tf = get_transforms(do_flip=True, flip_vert=False, max_rotate=30, max_zoom=0.1, max_lighting=0.1)

    return (ImageList.from_df(df_clf, path=path_train)
    .split_by_rand_pct(split)
    .label_from_df(cols=1)
    .transform(tf, size=size)
    .add_test(vision.Path(path_test).ls())
    .databunch(path=data_root, bs=bs)
    .normalize(vision.imagenet_stats))
            
data = get_data(bs=64, size=256)

# Create learner
learner = cnn_learner(data, models.resnet34, pretrained=True, metrics=accuracy, ps=0.5, callback_fns=ShowGraph)
learner.model_dir = "/kaggle/working" 
# Find optimal LR
learner.lr_find()
learner.recorder.plot()
# Train first epoch
learner.fit_one_cycle(1, max_lr=2e-2)
learner.show_results()
learner.save("resnet34_clf_256_1ep")
# Unfreeze the backbone and train second and third epochs 
# use same learning rate for the head
# use smaller learner rates for the backbone
learner.unfreeze()
learner.fit_one_cycle(2, max_lr=slice(1e-4, 2e-2))
learner.recorder.plot_lr(show_moms=True)
learner.save("resnet34_clf_256_ep3")
preds, _ = learner.get_preds(ds_type=DatasetType.Test)
preds = preds.argmax(dim=1)
preds = preds.numpy()
clf_out_df = pd.DataFrame(list(zip(map(lambda x: x.name, data.test_ds.items),preds)), columns=['ImageId','Label'])
clf_out_df.to_csv('clf_256_test_preds.csv', index=False)
learner.show_results()
from IPython.display import FileLink
FileLink('clf_256_test_preds.csv')
FileLink('resnet34_clf_256_ep3.pth')