import os



import pandas as pd

from matplotlib import pyplot as plt



from fastai import *

from fastai.vision import *



import json



%matplotlib inline
df_train = pd.read_csv("../input/iwildcam-desc/train_set.csv")



df_train = df_train[~df_train['annotations.category_id'].isnull()]

df_train['annotations.category_id'] = df_train['annotations.category_id'].astype(np.int64)



df_test = pd.read_csv("../input/iwildcam-desc/test_set.csv")
df_test
df_train.head()
train, test = [ImageList.from_df(df, path='../input/iwildcam2020-animal-crops/', cols='file_name', folder=folder, suffix='') 

               for df, folder in zip([df_train, df_test], ['animal_crops_train', 'animal_crops_test'])]

trfm = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,

                      p_affine=1., p_lighting=1.)

src = (train.use_partial_data(1)

        .split_from_df(col='is_valid')

        .label_from_df(cols='annotations.category_id')

        .add_test(test))

data = (src.transform(trfm, size=128, padding_mode = 'reflection')

        .databunch(path=Path('.'), bs=128).normalize(imagenet_stats))
print(data.classes)
def _plot(i,j,ax):

    x,y = data.train_ds[1]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
data.show_batch()
data.c
learn = cnn_learner(data, base_arch=models.resnet50, metrics=[accuracy])
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.recorder.min_grad_lr
learn.fit_one_cycle(12, slice(0.003))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(12, max_lr = slice(1e-5/2, 0.0005))
preds,y = learn.TTA(ds_type=DatasetType.Test)
learn.recorder.plot_losses()
org_classes = pd.DataFrame({"org_category": data.classes})

org_classes['Category'] = org_classes.index

org_classes.to_csv("category_mapping.csv")
pred_csv = pd.DataFrame(preds.numpy())

pred_csv['Id'] = learn.data.test_ds.items

pred_csv.to_csv("outout_preds.csv", index = False)