import os

GPU_id = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
import warnings

warnings.filterwarnings("ignore")



import fastai

print(fastai.__version__)

from fastai.vision import *

from fastai.callbacks import SaveModelCallback

import time
path = Path('../input/hymenoptera-data/hymenoptera_data')

print(type(path))

path.ls()
(path/'train').ls()
il = ImageList.from_folder(path)

il.items[0]
il
il[0].show()
sd = il.split_by_folder(train='train', valid='val')

sd
ll = sd.label_from_folder()

ll
%%time

x,y = ll.train[0]

x.show()

print(y,x.shape)
tfms = get_transforms(max_rotate=25); len(tfms)
ll = ll.transform(tfms,size=224)
%%time

x,y = ll.train[0]

x.show()

print(y,x.shape)
%%time

bs = 32

data = ll.databunch(bs=bs).normalize(imagenet_stats)
x,y = data.train_ds[0]

x.show()

print(y)
def _plot(i,j,ax): data.train_ds[0][0].show(ax)

plot_multi(_plot, 3, 3, figsize=(8,8))
xb,yb = data.one_batch()

print(xb.shape,yb.shape)

data.show_batch(rows=3, figsize=(10,8))
%%time

learn = cnn_learner(data, models.resnet18, metrics=accuracy)

learn.model_dir = '/kaggle/working/models'
!pwd
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,max_lr=slice(0.007),callbacks=[

            SaveModelCallback(learn, every='improvement', monitor='accuracy'),

            ])
pred, truth = learn.get_preds()
pred = pred.numpy()

truth = truth.numpy()

acc = np.mean(np.argmax(pred,axis=1) == truth)

print('Validation Accuracy %.4f'%acc)