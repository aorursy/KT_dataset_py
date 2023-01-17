from fastai.vision import * 

import warnings

warnings.filterwarnings('ignore')

from distutils.dir_util import copy_tree
classes = ['bananes', 'abricot', 'ananas', 'cerise', 'citron', 'clémentines', 'fraises', 'framboises', 'grenade', 'mango', 'orange', 'pamplemousse', 'pêches', 'poires', 'pomme']

#classes = ['pomme', 'orange']
copy_tree('../input/allfruits',"/kaggle/working/")



data = ImageDataBunch.from_folder(path="/kaggle/working/all", train="Train", test="Test", valid_pct=0.25, ds_tfms=get_transforms(), size=224, bs=16).normalize(imagenet_stats)
print(data)
learn = cnn_learner(data, models.resnet50, metrics=accuracy)


learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(10, slice(lr))
preds, y, losses= learn.get_preds(with_loss=True)
interp =ClassificationInterpretation(learn, preds, y, losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
""" pour montrer les images sur les quels il s'est le plus trompé"""

interp.plot_top_losses(9, figsize=(15,11), heatmap=True)
learn.save("trained_model", return_path=True)