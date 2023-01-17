%reload_ext autoreload 

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
bs = 64

np.random.seed(2)
data_path = "../input/poisonous_plants_dataset/"

data = ImageDataBunch.from_folder(data_path, bs=bs//2, size=299, ds_tfms=get_transforms(),num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=4,figsize=(8,8))
learn = create_cnn(data, models.resnet50, metrics=accuracy, path=".")
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8))
interp.plot_top_losses(9,figsize=(14,14))
interp.most_confused(min_val=0)
learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4,max_lr=slice(4e-6,1e-5))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8))
interp.plot_top_losses(9,figsize=(14,14))
test_data = ImageDataBunch.from_folder(data_path, bs=bs//2, size=299, ds_tfms=get_transforms(),num_workers=0,valid='test').normalize(imagenet_stats)

loss, acc = learn.validate(test_data.valid_dl)
print(f'Loss: {loss}, Accuracy: {acc*100} %')