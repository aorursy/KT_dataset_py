%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
bs=64

np.random.seed(2)
data_path = "../input/ancient_language_dataset"

data = ImageDataBunch.from_folder(data_path,ds_tfms=get_transforms(),size=224,bs=bs,num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
data.classes
learn = create_cnn(data, models.resnet34, metrics=accuracy, path=".")
learn.fit_one_cycle(4)
learn.save("stage-1")
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs=interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9,figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12),dpi=60)
interp.most_confused(min_val=1)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load("stage-1");
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2,max_lr=slice(7e-4,1e-3))
data = ImageDataBunch.from_folder(data_path,ds_tfms=get_transforms(),size=299,bs=bs//2,num_workers=0).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, path=".", metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(6,max_lr=slice(1e-5,1e-4))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.most_confused(min_val=0)