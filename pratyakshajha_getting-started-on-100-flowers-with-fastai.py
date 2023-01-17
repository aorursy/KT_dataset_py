import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



from fastai import *

from fastai.vision import *



%reload_ext autoreload

%autoreload 2

%matplotlib inline
df = pd.read_csv("../input/flower-goggle-tpu-classification/flowers_idx.csv")

label = pd.read_csv("../input/flower-goggle-tpu-classification/flowers_label.csv")
df.head()
label.head()
path = Path('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/flowers_google/')

path_test = Path('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/')
tfms = get_transforms(do_flip=True,max_rotate=0.1,max_lighting=0.15)

test = (ImageList.from_folder(path_test,extensions='.jpeg'))



data = (ImageList.from_df(df,path,folder='flowers_google',suffix='.jpeg',cols='id')

                .split_by_rand_pct(0.15)

                .label_from_df(cols='flower_cls')

                .transform(tfms)

                .add_test(test)

                .databunch(bs=32)

                .normalize(imagenet_stats))
data.show_batch(rows=4)
arch = models.resnet50

learn = cnn_learner(data, arch, metrics=accuracy, model_dir='/kaggle/working')
learn.lr_find()

learn.recorder.plot()
learn.summary()
lr = 1e-2
learn.fit_one_cycle(6,lr,moms=(0.9,0.8))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(12,figsize=(20,8))
interp.most_confused(min_val=3)
img = open_image('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/d9cb87ad0.jpeg')

print(learn.predict(img)[0])

img
samp = pd.read_csv('/kaggle/input/flower-classification-with-tpus/sample_submission.csv')

n = samp.shape[0]

path_alltest = '/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/'
for i in range(n):

    idc = samp.iloc[i][0]

    k = path_alltest + idc + '.jpeg'

    k = open_image(k)

    ans = learn.predict(k)[0]

    samp.loc[[i],1:] = str(ans)
samp.head(10)
lab = {}

for i in range(label.shape[0]):

  sha = label.iloc[i]

  lab[sha[1]]=int(sha[0])
samp.label.replace(lab,inplace=True)
samp.head()
samp.to_csv('submission.csv',index=False)