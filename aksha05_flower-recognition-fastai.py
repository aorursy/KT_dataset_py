from fastai.vision import *

from fastai.metrics import error_rate

import os
bs = 64
path = '/kaggle/input/flower-image-dataset/flowers'
os.listdir(path)
fnames = get_image_files(path)  #extract file names

fnames[:5] 
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$' ## regex which will extract the labels from the filenames
data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms = get_transforms(), size = 224, bs = bs

                                  ).normalize(imagenet_stats)  
print(data.classes) # different classes of flowers
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4) ### train for 4 epochs 
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()
interp.plot_top_losses(8, figsize = (12,8)) ## these are the top losses - predicted vs actual
interp.plot_confusion_matrix(figsize=(12,12), dpi=60) # confusion matrix for the losses