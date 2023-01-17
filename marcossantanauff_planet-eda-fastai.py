%reload_ext autoreload

%autoreload 2

%matplotlib inline



import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from fastai.vision import *
PATH = Path('../input/planet-understanding-the-amazon-from-space')

TRAIN = Path('../input/planet-understanding-the-amazon-from-space/train-jpg')

TEST = Path('../input/planet-understanding-the-amazon-from-space/test-jpg-v2')

PATH.ls()
df = pd.read_csv(PATH/'train_v2.csv')

samplesub = pd.read_csv(PATH/'sample_submission_v2.csv')
df.head()
print('Number of training files = {}'.format(len(df)))

print('Number of test files = {}'.format(len(samplesub)))



print('Number of training files = {}'.format(len(TRAIN.ls())))

print('Number of test files = {}'.format(len(TEST.ls())))
labels = df.groupby('tags')['image_name'].count().reset_index()

labels.head()
labels.sort_values('image_name',ascending=False).head()
#sns.barplot(x=labels['tags'],y=labels['image_name'])

import matplotlib.ticker as ticker

plt.figure(figsize=(30,12))

ax = sns.barplot(x='tags',y='image_name',data=labels)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
sample_primary = df.loc[df['tags']=='clear primary'].head()

sample_partly_cloudy = df.loc[df['tags']=='partly_cloudy primary'].head()
sample_partly_cloudy
sample_primary
open_image(TRAIN/'train_2.jpg') # Clear primary

open_image(TRAIN/'train_17.jpg') # partly_cloudy primary
#tfms = [[*rand_resize_crop(256),dihedral(),zoom(scale=1.05)],[]]

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
src = ImageList.from_df(df,path=TRAIN,cols='image_name',suffix='.jpg').split_by_rand_pct(0.2).label_from_df(cols='tags',label_delim=' ')
data = src.transform(tfms).databunch(bs=64).normalize(imagenet_stats)
data.show_batch(rows=3)
arch = models.resnet50
learn = cnn_learner(data,arch,metrics=[fbeta],model_dir='/kaggle/working')
learn.lr_find()

# Find a good learning rate
learn.recorder.plot()
lr = 1e-2
learn.fit_one_cycle(7,slice(lr))
learn.save('stage1-256-resnet50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(7,slice(1e-5,lr/5))
learn.recorder.plot_losses()
learn.save('stage2-256-resnet50')
learn.export('/kaggle/working/export.pkl')
test = ImageList.from_folder(TEST).add(ImageList.from_folder(PATH/'test-jpg-additional'))

len(test)
learn = load_learner(Path('/kaggle/working'), test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
thresh = 0.5

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
labelled_preds[:5]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df_preds = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df_preds.to_csv('/kaggle/working/submission.csv', index=False)
a = df_preds.sort_values('image_name',ascending=True)

a.head()
df_preds.shape
samplesub.tail(50)
samplesub.shape
#! kaggle competitions submit planet-understanding-the-amazon-from-space -f {'/kaggle/working/submission.csv'} -m "My submission"