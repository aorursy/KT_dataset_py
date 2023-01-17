%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64 #batch size 
path = untar_data(URLs.PETS);path # download data set from s3
path.ls()
path_anno = path/'annotations' # folder containes annotations

path_imgs = path/'images' # folder containes images 
fnames = get_image_files(path_imgs) #get all file names from folder 
fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$' # get feature from the folder path
data = ImageDataBunch.from_name_re(path_imgs, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
data.show_batch(3, figsize=(7,6))
data.classes
len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
intrep = ClassificationInterpretation.from_learner(learn)
losses, idxs = intrep.top_losses()
len(data.valid_ds) == len(losses) == len(idxs)
intrep.plot_top_losses(9,figsize=(16,10))
doc(intrep.plot_top_losses)
intrep.plot_confusion_matrix(figsize=(12,12),dpi=60)
intrep.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2,max_lr=slice(1e-6,1e-4))