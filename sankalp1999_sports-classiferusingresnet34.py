%reload_ext autoreload

%autoreload 2

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
from fastai import *

from fastai.vision import *
bs = 32
path = '/kaggle/input/dataset'
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size = 224, num_workers = 0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(24,24))
print(data.classes)
len(data.classes), data.c
learner = cnn_learner(data, models.resnet34, model_dir = '/tmp/models' ,metrics=[accuracy])

learner.lr_find()

learner.recorder.plot()
learner.save('Res34stage1')
learner.unfreeze()

learner.fit_one_cycle(8, 1e-3)
learner.load('Res34stage1')
learner.unfreeze()

learner.fit_one_cycle(15, max_lr = slice(1e-4, 1e-3)) 
# learner.save('Res34BestTillNow')
np.random.seed(2)

learner.load('Res34stage1')

learner.fit_one_cycle(10, max_lr = slice(1e-4, 1e-3)) 
np.random.seed(2)

big_learner = cnn_learner(data, models.resnet50, model_dir = '/tmp/models' ,metrics=[accuracy])
# big_learner.save('Res50stage1')


big_learner.unfreeze()

big_learner.fit_one_cycle(7, max_lr = slice(1e-4, 1e-3)) 
big_learner.save('Res50BestTillNow') # 85.7143 percent
np.random.seed(2)

big_learner = cnn_learner(data, models.resnet50, model_dir = '/tmp/models' ,metrics=[accuracy])

big_learner.unfreeze()

big_learner.fit_one_cycle(7, max_lr = slice(1e-4, 1e-3)) 
big_learner.save('Res50Best88.9') # 85.7143 percent
interp = ClassificationInterpretation.from_learner(big_learner)
interp.plot_top_losses(9, figsize = (15,11))