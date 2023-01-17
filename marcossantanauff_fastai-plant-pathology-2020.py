%matplotlib inline

import numpy as np

import pandas as pd

from fastai.vision import *

import matplotlib.pyplot as plt

import seaborn as sns

import os

import shutil

path = Path('../input/plant-pathology-2020-fgvc7/images')
path.ls()[0:3]
outpath = Path('/kaggle/working')

outpath.ls()
ii = open_image(path/'Train_1767.jpg')

ii.show()

train = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
classes=['healthy', 'multiple_diseases', 'rust', 'scab']
train.head(5)
# https://www.kaggle.com/otzhora/fastai-simple-efficientnet-ensemble-solution

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

    train_labels = train_labels.drop(columns=['healthy', 'multiple_diseases', 'rust', 'scab'])

    return train_labels
train_tag = transform_data(train)
train_tag
rust = train_tag[train_tag['tag']=='rust']
rust.head()
rust_leaves = list(rust['image_id']);len(rust_leaves)
_,axs = plt.subplots(1,3,figsize=(22,22))

open_image(path/'Train_3.jpg').show(ax=axs[0],title='1')

open_image(path/'Train_10.jpg').show(ax=axs[1],title='2')

open_image(path/'Train_15.jpg').show(ax=axs[2],title='3')
scab = train_tag[train_tag['tag']=='scab']

scab.head(10)
scab_leaves = list(scab['image_id']);len(scab_leaves)
sc = open_image(path/'Train_0.jpg')

sc
multi_d = train_tag[train_tag['tag']=='multiple_diseases']

multi_d.head(10)
multi_leaves =  list(multi_d['image_id'])
mt1 = open_image(path/'Train_122.jpg')

mt2 = open_image(path/'Train_113.jpg')

mt3 = open_image(path/'Train_95.jpg')

_,axs = plt.subplots(1,3,figsize=(22,22))

mt1.show(ax=axs[0],title='1')

mt2.show(ax=axs[1],title='2')

mt3.show(ax=axs[2],title='3')
ht = train_tag[train_tag['tag']=='healthy']

ht.head(10)
healthy_leaves = list(ht['image_id'])

healthy_leaves[0:3]
ht1 = open_image(path/'Train_2.jpg')

ht2 = open_image(path/'Train_4.jpg')

ht3 = open_image(path/'Train_5.jpg')

_,axs = plt.subplots(1,3,figsize=(22,22))

ht1.show(ax=axs[0],title='1')

ht2.show(ax=axs[1],title='2')

ht3.show(ax=axs[2],title='3')
print(len(scab),len(rust),len(multi_d),len(ht))
dd = train_tag.groupby('tag')['image_id'].count().reset_index()
dd
sns.barplot(x='tag', y='image_id', data=dd)
# https://www.kaggle.com/lextoumbourou/plant-pathology-2020-eda-training-fastai2

_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))

for ax, column in zip(axes, classes):

    train[column].value_counts().plot.bar(title=column,ax=ax)



plt.show()
from sklearn.model_selection import train_test_split
trainsplit,validsplit = train_test_split(train_tag,test_size=0.30,random_state=42,stratify=train_tag['tag'])
trainsplit.shape,validsplit.shape
trainsplitdd = trainsplit.groupby('tag')['image_id'].count().reset_index()
trainsplitdd
validplitdd = validsplit.groupby('tag')['image_id'].count().reset_index()
validplitdd
sns.barplot(x='tag', y='image_id', data=trainsplitdd)
valid_idx = validsplit.index

train_idx = trainsplit.index
tfms = get_transforms()
np.random.seed(42)

src = ImageList.from_df(path=path,df=train_tag).split_by_idx(valid_idx).label_from_df('tag')
src
data_512 = src.transform(tfms,size=512).databunch(bs=8).normalize(imagenet_stats)

#data_1020 = src.transform(tfms,size=1020).databunch(bs=8).normalize(imagenet_stats)
data_512,data_512.show_batch(5,figsize=(7,7)),data_512.classes,data_512.c
learn = cnn_learner(data_512,models.densenet169,wd=1e-4,metrics=accuracy,model_dir=outpath)
learn.lr_find(num_it=300)
learn.recorder.plot(suggestion=True)
lr = 2e-3

learn.fit_one_cycle(8,lr)
learn.recorder.plot_losses()
learn.save('stage1_512')
(outpath).ls()
learn.load('stage1_512')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3,slice(8e-6,1e-4))
learn.save('stage2_512')
#data_1020 = src.transform(tfms,size=1020).databunch(bs=8).normalize(imagenet_stats)



#learn.data = data_1020

#data_1020.train_ds[0][0].shape
#learn.freeze()
#learn.lr_find()
#learn.recorder.plot()
#learn.fit_one_cycle(5, slice(lr))
#learn.save('stage1_1020')




#learn.unfreeze()



#learn.fit_one_cycle(5, slice(1e-5, lr/5))
#learn.save('stage2_1020')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(k=16,figsize=(20,20),heatmap=False)
learn.load('stage2_512')
learn.export(outpath/'export.pkl')
outpath.ls()
test_images = ImageList.from_folder(path)

test_images.filter_by_func(lambda x: x.name.startswith("Test"))
test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
test_df.image_id = [image_id+'.jpg' for image_id in test_df.image_id]
test_df.head()
learn = load_learner(outpath)
all_test_preds = []

from tqdm import tqdm_notebook as tqdm

for item in tqdm(test_images.items):

    name = item.name[:-4]

    img = open_image(item)

    preds = learn.predict(img)[2]

    all_test_preds.append(preds)

   # test_df.loc[name]['healthy'] = preds[0]

   # test_df.loc[name]['multiple_diseases'] = preds[1]

   # test_df.loc[name]['rust'] = preds[2]

  #  test_df.loc[name]['scab'] = preds[3]
aa = [f.numpy() for f in all_test_preds]
bb = np.stack(aa,axis=0)

len(bb)
bb
test_df_output = pd.concat([test_df, pd.DataFrame(bb, columns=classes)], axis=1).round(6)
test_df_output.head()
test_df_output['image_id'] = test_df['image_id'].str.strip('.jpg')
test_df_output.to_csv('/kaggle/working/submission3.csv',index=False)
data.save(outpath/'data.pkl')
outpath.ls()