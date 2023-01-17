from fastai.utils import *

from fastai.vision import *

from fastai.callbacks import *

from pathlib import Path

import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import PIL

from torch.utils import model_zoo



%matplotlib inline
!ls ../input
data_path = Path("../input/")

train_df = pd.read_csv(data_path/'train.csv')

test_df = pd.read_csv(data_path/'sample_submission.csv')
sns.countplot('has_cactus', data=train_df)

plt.title('Classes', fontsize=15)

plt.show()
plt.figure(figsize=(8,6))



i = 0

sample = train_df.sample(12)

for row in sample.iterrows():

    img_name = row[1][0]

    img = PIL.Image.open(data_path/'train'/'train'/img_name)

    i += 1

    plt.subplot(3,4,i)

    title = 'Not Cactus (0)'

    if row[1][1] == 1:

        title = 'Cactus (1)'

    plt.title(title, fontsize=10)

    plt.imshow(img)

    plt.axis('off')



plt.subplots_adjust(top=0.90)

plt.suptitle('Sample of Images', fontsize=16)

plt.show()
df1 = train_df[train_df.has_cactus==0].copy()

df2 = df1.copy()

train_df = train_df.append([df1, df2], ignore_index=True)
sns.countplot('has_cactus', data=train_df)

plt.title('Oversampled Classes', fontsize=15)

plt.show()
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=20, max_lighting=0.3, max_warp=0.2, max_zoom=1.2)
test_images = ImageList.from_df(test_df, path=data_path/'test', folder='test')



src = (ImageList.from_df(train_df, path=data_path/'train', folder='train')

       .split_by_rand_pct(0.2)

       .label_from_df()

       .add_test(test_images))
data = (src.transform(tfms, 

                     size=128,

                     resize_method=ResizeMethod.PAD, 

                     padding_mode='reflection')

        .databunch(bs=256)

        .normalize(imagenet_stats))
data.classes, data.c
data.show_batch(rows=3, figsize=(9,9))


learn = cnn_learner(data,

                    models.resnet101,pretrained=True,

                    metrics=[accuracy, AUROC()],path='',model_dir='work')
!ls
learn.lr_find(stop_div=True)

learn.recorder.plot(suggestion=True)
lr = 3.98E-04

learn.fit_one_cycle(5, lr)
learn.recorder.plot_losses()
learn.recorder.plot_lr(show_moms=True)
learn.save('step-1')
!ls work
learn = cnn_learner(data,

                    models.resnet101, pretrained=True,

                    metrics=[accuracy, AUROC()], 

                    callback_fns=[partial(SaveModelCallback)],

                    wd=0.1,

                    ps=[0.9, 0.6, 0.4],

                    path = '')

learn = learn.load('../work/step-1')
learn.freeze_to(4)

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3)
learn.fit_one_cycle(7)
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)



interp.plot_confusion_matrix(figsize=(2,2))
interp.plot_top_losses(4, figsize=(6,6))
probability, classification = learn.get_preds(ds_type=DatasetType.Test)

test_df.has_cactus = probability.numpy()[:, 0]

test_df.head()
test_df.to_csv("submission.csv", index=False)