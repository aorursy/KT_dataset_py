# We import required lib's

from fastai.vision import *

import os
# nothing in our folder yet

print(os.listdir("../working"))
# move images to working folder

!mkdir ../working/dog

!mkdir ../working/wolf

!cp ../input/dog-v1/* ../working/dog

!cp ../input/wolf-v1/* ../working/wolf
# let's check

print(os.listdir("../working"))
verify_images('../working/wolf', delete=True, max_size=500)

verify_images('../working/dog', delete=True, max_size=500)
classes = ['wolf','dog']

np.random.seed(42)

data = ImageDataBunch.from_folder('../working', train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(15,11))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
data.show_batch(rows=3, figsize=(15,11))