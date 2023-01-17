!wget https://static.leiphone.com/car.zip
!unzip -qq car.zip
!mkdir -pv data

!mv -v train test train.csv data
import os

import random

from shutil import copyfile



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt
df_label = pd.read_csv('data/train.csv')



os.mkdir('images')

os.mkdir('images/train')

os.mkdir('images/train/0')

os.mkdir('images/train/1')

os.mkdir('images/train/2')



for _, (filename, label) in df_label.iterrows():

    copyfile(f'data/train/{filename}', f'images/train/{label}/{filename}')

    

os.mkdir('images/valid')

os.mkdir('images/valid/0')

os.mkdir('images/valid/1')

os.mkdir('images/valid/2')



for label in [0, 1, 2]:

    img_list = os.listdir(f'images/train/{label}')

    random.shuffle(img_list)

    to_move = img_list[:344] 

    for filename in to_move:

        os.rename(f'images/train/{label}/{filename}', f'images/valid/{label}/{filename}')

        

os.mkdir('images/test')

for filename in os.listdir('data/test'):

    copyfile(f'data/test/{filename}', f'images/test/{filename}')

    

os.mkdir('submissions')
from fastai import *

from fastai.vision import *
data = ImageDataBunch.from_folder(path="./images", 

                                  train="train", 

                                  valid="valid", 

                                  test="test", 

                                  ds_tfms=get_transforms(max_zoom=1.25), 

                                  size=224, 

                                  bs=64)

data.normalize(imagenet_stats)
learn = cnn_learner(data, models.densenet121, metrics=[accuracy])
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(30, 1e-02)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
predictions, targets = learn.TTA(ds_type=DatasetType.Test)
classes = predictions.argmax(1)
test_images = [i.name for i in learn.data.test_ds.items][:9]



plt.figure(figsize=(10,8))

for i, fn in enumerate(test_images):

    img = plt.imread("./images/test/" + fn, 0)

    plt.subplot(3, 3, i+1)

    plt.imshow(img)

    plt.title(f'{fn}: {classes[i]}')

    plt.axis("off")
df = pd.DataFrame()



df['filename'] = [i.name for i in learn.data.test_ds.items]

df['prediction'] = classes

df['id'] = df.filename.apply(lambda x: x.replace('.jpg', '')).astype('int64')



submission = df[['id', 'prediction']]

submission = submission.sort_values(by='id')

submission.head()
submission.to_csv('submission.csv', index=False, header=None)
!rm -r car.zip data images