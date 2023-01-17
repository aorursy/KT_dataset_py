%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *



from PIL import Image

from matplotlib.pyplot import imshow



from random import randint



from tqdm import tqdm



import os
import os

print(os.listdir("../input/"))
PATH = "../input/fashionmnistimages/fashion-mnist/data/"
sz=28

bs=256
tfms = get_transforms()

data = ImageDataBunch.from_folder(PATH, 

    ds_tfms=tfms, size=sz,bs=bs, num_workers=0).normalize(imagenet_stats)
print(f'We have {len(data.classes)} different types of clothing\n')

print(f'Types: \n {data.classes}')
data.show_batch(8, figsize=(20,15))
from os.path import expanduser, join, exists

from os import makedirs

cache_dir = expanduser(join('~', '.torch'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)



# copy time!

!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir='/output/model/',callback_fns=ShowGraph)
lrf=learn.lr_find()

learn.recorder.plot()
lr=1e-2
learn.fit_one_cycle(1,lr)
learn.save('zalando-stage-1')
learn.unfreeze()
lrf=learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, max_lr=slice(5e-6, 5e-4))
learn.save('zalando-stage-2')
learn.fit_one_cycle(6, max_lr=slice(5e-6, 5e-4))
learn.save('zalando-stage-3')
interp = ClassificationInterpretation.from_learner(learn,tta=True)
interp.plot_top_losses(16, figsize=(20,14))
interp.plot_confusion_matrix(figsize=(12,12), dpi=100, normalize=True, norm_dec=0, cmap=plt.cm.YlGn)
def accuracy_topk(output, target, topk=(3,)):

  """Computes the precision@k for the specified values of k"""

   

  maxk = max(topk)

  batch_size = target.size(0)

   

  _, pred = output.topk(maxk, 1, True, True)

  pred = pred.t()

  correct = pred.eq(target.view(1, -1).expand_as(pred))

 

  res = []

  for k in topk:

    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

    res.append(correct_k.mul_(100.0 / batch_size))

  return res
output, target = learn.get_preds(ds_type=DatasetType.Valid)

output_tta, target_tta = learn.TTA(ds_type=DatasetType.Valid)
accuracy = [accuracy_topk(output=output, target=target,topk=(i,)) for i in range(10)]

accuracy_tta = [accuracy_topk(output=output_tta, target=target_tta,topk=(i,)) for i in range(10)]

plt.plot(accuracy)

plt.plot(accuracy_tta)