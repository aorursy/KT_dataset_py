%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
bs = 16
path = Path('../input/jaguar_vs_leopard/')
path.ls()
np.random.seed(12)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=0).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(rows=3, figsize=(3,4))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/kaggle/working/")
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.export('/kaggle/working/export.pkl')
learn = load_learner('/kaggle/working/')
img = open_image(path/'leopard'/'Z.jpg')
img
pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)
img2 = open_image(path/'jaguar'/'Z.jpg')
img2
pred_class,pred_idx,outputs = learn.predict(img2)
print(pred_class)