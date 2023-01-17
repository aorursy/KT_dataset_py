%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
#help(untar_data)

pets = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet"
path = untar_data(pets); path
path.ls()
path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)

data.normalize(imagenet_stats)
help(data.show_batch)
data.show_batch(rows=3)
print(data.classes)
len(data.classes)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9)
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi =60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))