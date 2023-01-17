# Import linraries



import os

print(os.listdir("../input"))



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *
# define image data directory path

DATA_DIR='../input/data'
# The directory under the path is the label name.

os.listdir(f'{DATA_DIR}')
# Check if GPU is available

torch.cuda.is_available()
# create image data bunch

data = ImageDataBunch.from_folder(DATA_DIR, 

                                  train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,

                                  bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
# check classes

print(f'Classes: \n {data.classes}')
# show some sample images



data.show_batch(rows=3, figsize=(7,6))
# build model (use resnet34)

learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
# search appropriate learning rate

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6,1e-2)
# save stage

learn.save('stage-1')
learn.unfreeze()
# search appropriate learning rate

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5 ))
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5 ))
# save stage

learn.save('stage-2')
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)