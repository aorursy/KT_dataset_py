from fastai.vision import *
from fastai.widgets import DatasetFormatter
from fastai.widgets import ImageCleaner

from pathlib import Path

import pandas as pd
import numpy as np

import shutil
import os

pd.set_option("display.max_rows", None, "display.max_columns", None)
classes = {'man': '../input/manvswoman/man_download.csv',
           'woman': '../input/manvswoman/woman_download.csv'
           }
path = Path('data/')

for label in classes:
    dest = path/label
    print(path/label)
    dest.mkdir(parents=True, exist_ok=True)
    classes[label]
    download_images(classes[label], dest, max_pics=200)
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.3,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-05,1e-04))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix()
n_mistakes = 15
interp.plot_top_losses(n_mistakes, figsize=(15,11))
db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
ImageCleaner(ds, idxs, path)
new_classes = pd.read_csv('./data/cleaned.csv')
new_classes.head()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
root_path = '../working/data/{0}'
img=mpimg.imread(root_path.format('woman') + '/00000002.jpg')
imgplot = plt.imshow(img)
plt.show()
path = Path('clean_data/')

classes = new_classes['label'].unique()

for label in classes:
    folder_name = label
    dest = path/folder_name
    dest.mkdir(parents=True, exist_ok=True)    
counter = 0

for label in classes:
    root_path = '../working/data/{0}'
    dest_path = '../working/clean_data/{0}'
    temp_df = new_classes.loc[new_classes['label'] == label]    
    files = list(temp_df['name'])
    for f in files:
        img_origin = '/'.join([root_path.format(label), f.split('/')[1]])
        img_dest = '/'.join([dest_path.format(label), str(counter) + '.jpg'])
        shutil.copy(img_origin, img_dest)
        counter += 1
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.3,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
new_learn = cnn_learner(data, models.resnet34, metrics=error_rate)
new_learn.fit_one_cycle(4)
new_learn.save('stage-3')
interp = ClassificationInterpretation.from_learner(new_learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix()
n_mistakes = 3
interp.plot_top_losses(n_mistakes, figsize=(15,11))
learn.export()