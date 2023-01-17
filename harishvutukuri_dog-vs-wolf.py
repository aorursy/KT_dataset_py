from fastai.vision import *



import numpy as np



import os

os.chdir('/kaggle/input/dogs-vs-wolves/data')
np.random.seed(42)



path = Path('.')



data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(4)
learn.model_dir='/kaggle/working/'



learn.save('stage-1')

learn.unfreeze()
learn.lr_find()

learn.recorder.plot(skip_start=0, skip_end=0)
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))
learn.save('stage-2')

learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()