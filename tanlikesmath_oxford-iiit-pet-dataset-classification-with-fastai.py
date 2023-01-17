%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
import os

cwd = os.getcwd()

print(cwd)

path_img = Path('../input/images/images')
fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2) #random seed for reproducibility

pat = r'/([^/]+)_\d+.jpg$' #regex expression
sz = 128

bs = 64
data = ImageDataBunch.from_name_re('.', fnames, pat, ds_tfms=get_transforms(), size=sz, bs=bs

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
from fastai.metrics import accuracy

learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=1e-2)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)