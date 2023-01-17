%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
import fastai

print('fastai version :', fastai.__version__)
# Batch Size (adequate size for GPU of 11GB or more)

bs = 64
path = Path('../input/planesnet/planesnet/planesnet')

# path.ls()
fnames = get_image_files(path)

fnames[:5]
# regex to extract category

pat = r'^\D*(\d+)'
np.random.seed(23)
# Setup the transformations to apply to the training data

tfms = get_transforms(flip_vert=True)



# Add the images to the Image Data Bunch

data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=tfms, size=21, bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/output/kaggle/working/model")
learn.model
learn.fit_one_cycle(4)
learn.save('stage-1', return_path=True)
interp = ClassificationInterpretation.from_learner(learn)



losses, idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(6, slice(1e-5,1e-3))
learn.save('stage-2', return_path=True)
interp = ClassificationInterpretation.from_learner(learn)



losses, idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)