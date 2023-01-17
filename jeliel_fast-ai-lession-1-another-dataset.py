from fastai import *

from fastai.vision import *
ds = 64
path = untar_data(URLs.MNIST); path
path.ls()
path_training = path/'training'

path_testing = path/'testing'
data = ImageDataBunch.from_folder(path, tain='training', test='testing', valid_pct= 0.2, size=16)
data.show_batch(3, figsize=(5,5))
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.save('stage-2')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-5))
learn.save('stage-2')