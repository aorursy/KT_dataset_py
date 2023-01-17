# !kaggle competitions download -c virtual-hack
# !unzip -q car_data.zip
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

torch.manual_seed(42)

torch.cuda.manual_seed_all(42)
bs = 64

# !ls 'car_data/train'
path = Path('../input/car_data/car_data')
path.ls()
data = ImageDataBunch.from_folder(path, train="train", valid_pct=0.3,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(8,7))
# print(data.classes)

len(data.classes),data.c
pwd
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/kaggle/working', callback_fns=ShowGraph)
learn.model
learn.fit_one_cycle(10)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(), 

                                  valid='test', size=224, bs = 64) .normalize(imagenet_stats)
learn.load('stage-1');

# learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-3))
learn.save('stage-2')
learn.load('stage-2');
learn.unfreeze()
data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.), 

                                  valid='test', size=224, bs = 64) .normalize(imagenet_stats)
get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
learn.lr_find()
# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, max_lr=1e-4)
data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(), 

                                  valid='test', size=229, bs = 64) .normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/kaggle/working', callback_fns=ShowGraph)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=1e-2)
learn.save('stage-1-50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
torch.manual_seed(42)

torch.cuda.manual_seed_all(42)

learn.fit_one_cycle(10, max_lr=slice(1e-5,1e-4))
learn.load('stage-1-50');



interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
# interp.plot_confusion_matrix(figsize=(12,12),cmap='viridis', dpi=60)
data = ImageDataBunch.from_folder(path, train="train", ds_tfms=get_transforms(), 

                                  valid='test', size=224, bs = 64) .normalize(imagenet_stats)

learn = cnn_learner(data, models.densenet169, metrics=accuracy, model_dir='/kaggle/working', callback_fns=ShowGraph)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(20, max_lr=1e-2)