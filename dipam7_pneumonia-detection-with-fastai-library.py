path = "../input/chest_xray/chest_xray/"

print(path)
from fastai import *

from fastai.vision import *
# set the batch size i.e. the number of images to train at a time

# reduce this number if you get an out of memory error 

bs = 64
# create a data bunch

np.random.seed(42)

data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(do_flip=False), 

                                  size=224, num_workers=0, 

                                  bs=bs, valid_pct=0.2).normalize(imagenet_stats)
# display 3 rows of data

data.show_batch(rows=3, figsize=(7,6))
# verify the classes

print(data.classes)

len(data.classes),data.c
# create a neural network

learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
# fit 4 layers

learn.fit_one_cycle(4)
# interpret the results

interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
# plot the top losses

interp.plot_top_losses(9, figsize=(15,11))
# plot the confusion matrix

interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()
learn.recorder.plot()
# we select the slice with the steepest slope

learn.unfreeze()

learn.fit_one_cycle(1, max_lr=slice(1e-3,1e-2))
# interpret the results

interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)