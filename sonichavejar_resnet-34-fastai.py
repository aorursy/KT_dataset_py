from fastai import *

from fastai.vision import *



%matplotlib inline
path = '/kaggle/input/simpsons_dataset/simpsons_dataset/'

bs = 256
fnames = get_image_files(path)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), valid_pct=0.2, size=224, num_workers=4, bs=bs).normalize()
data.show_batch(3)
learn = cnn_learner(data, models.resnet34, metrics=accuracy)

learn.unfreeze()
learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12, 12))