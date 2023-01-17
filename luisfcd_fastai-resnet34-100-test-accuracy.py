import fastai

from fastai.vision import *
data = ImageDataBunch.from_folder(path = '/kaggle/input/10-monkey-species/', train = 'training', 

                                  valid = 'validation', 

                                  size=256,bs=64, ds_tfms=get_transforms()).normalize(imagenet_stats)

data
data.show_batch(3, figsize=(10,7));

learn = cnn_learner(data, models.resnet34, 

                   metrics=[accuracy, AUROC()], model_dir='/kaggle/working/models')
learn.lr_find()
learn.recorder.plot(suggestion= True)
learn.fit_one_cycle(6, 1.45E-03)
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-5, 1e-4))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_top_losses(36, figsize=(20,20), heatmap = True)
interp.plot_confusion_matrix()