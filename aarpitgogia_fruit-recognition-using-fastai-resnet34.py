%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai import *

from fastai.vision import *

np.random.seed(42)
path = URLs.LOCAL_PATH/'../input/fruits-360_dataset/fruits-360'

path.ls()
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder((path), train="Training", valid="Test", ds_tfms=tfms, size=52)
learn = create_cnn(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(8, max_lr=slice(1e-2, 1e-1))