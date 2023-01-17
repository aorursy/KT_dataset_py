%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import *
!pwd
print(os.listdir('../input'))
path = Path('../input/messy-vs-clean-room/images'); path
path.ls()
path_train = path/'train'

path_valid = path/'val'

path_test = path/'test'
tfs = get_transforms()
data = ImageDataBunch.from_folder(path, valid='val', test='test', size=224, bs=48)
data.normalize(imagenet_stats)
data.show_batch(rows=2, figsize=(7, 6))
# see data length

print(data.classes)

len(data.classes), data.c
learn = cnn_learner(data, models.resnet34, metrics= [error_rate, accuracy])
# applying a learning technique

learn.fit_one_cycle(4)
learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-5, 1e-4))
interp = learn.interpret()
losses, idx = interp.top_losses()
len(data.valid_ds) == len(losses) == len(idx) 
interp.plot_top_losses(9, figsize=(15,6))
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)