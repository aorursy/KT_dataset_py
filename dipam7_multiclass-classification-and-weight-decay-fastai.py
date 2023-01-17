from fastai import *

from fastai.vision import *
bs = 128
path = "../input/nonsegmentedv2/"
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(do_flip=False), 

                                  size=224, num_workers=0, 

                                  bs=bs, valid_pct=0.2).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/")
learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(5,5), dpi=120)
learn.save("/kaggle/working/non-wd-stage-1")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))
learn.save("/kaggle/working/non-wd-stage-2")
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save("/kaggle/working/non-wd-stage-3")
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/", wd=1e-1)
learn.fit_one_cycle(5)
learn.save("/kaggle/working/wd-stage-1")
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/", wd=10)
learn.fit_one_cycle(5)
learn.save("/kaggle/working/large-wd-stage-1")
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))
learn.save("/kaggle/working/large-wd-stage-2")