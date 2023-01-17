%reload_ext autoreload

%autoreload 2

%matplotlib inline
from pathlib import PosixPath

path = PosixPath('../input')
import pandas as pd

df = pd.read_csv(path/'train.csv')
df.id = 'train/train/' + df.id
df.head()
from fastai.vision import *

from fastai.metrics import error_rate
src = (ImageList.from_df(df, path)

       .split_by_rand_pct(0.2)

       .label_from_df(1))
tfms=get_transforms()

data = (src.transform(tfms, size=32)

        .databunch()

        .normalize(imagenet_stats))
data.train_ds[0][0].shape
data.show_batch(rows=3, figsize=(9,7))
data.classes, data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
learn.data.train_ds[0][0].shape
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8, slice(1e-2))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(9,7))
interp.plot_confusion_matrix()
learn.recorder.plot_losses()
learn.recorder.plot_lr()
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, slice(1e-5/2))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(9,7))
interp.plot_confusion_matrix()
learn.recorder.plot_losses()
learn.recorder.plot_lr()
learn.save('stage-2')
tfms=get_transforms(flip_vert=True)

data64 = (src.transform(tfms, size=64)

          .databunch()

          .normalize(imagenet_stats))
learn.data = data64
learn.freeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, slice(1e-2))
learn.fit_one_cycle(4, slice(1e-2))
learn.fit_one_cycle(4, slice(1e-2))
learn.fit_one_cycle(4, slice(1e-2))
learn.fit_one_cycle(4, slice(1e-2))
learn.fit_one_cycle(4, slice(1e-2))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(9,7))
interp.plot_confusion_matrix()
learn.recorder.plot_losses()
learn.recorder.plot_lr()
learn.save('stage-3')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, slice(1e-5, 1e-2/5))
learn.fit_one_cycle(4, slice(1e-5, 1e-2/5))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(9,7))
interp.plot_confusion_matrix()
learn.recorder.plot_losses()
learn.recorder.plot_lr()
learn.save('stage-4')
learn.export("/export.pkl")
predictor = load_learner("/", test=ImageList.from_df(df, path))
preds_train, y_train, losses_train  = predictor.get_preds(ds_type=DatasetType.Test, with_loss=True)

preds_train[:5], y_train[:5], losses_train[:5]
y_train = torch.argmax(preds_train, dim=1)
interp = ClassificationInterpretation(predictor, preds_train, tensor(df.has_cactus.values), losses_train)
interp.plot_confusion_matrix()
from sklearn.metrics import roc_auc_score

def roc_auc(y_pred, y_true):

    return roc_auc_score(y_true, y_pred)
roc_auc(y_train, df.has_cactus.values)
predictor = load_learner("/", test=ImageList.from_folder(path/'test/test'))
preds_test, y_test, losses_test  = predictor.get_preds(ds_type=DatasetType.Test, with_loss=True)

preds_test[:5], y_test[:5], losses_test[:5]
y_test = torch.argmax(preds_test, dim=1)

y_test
sub_df = pd.DataFrame({'id': os.listdir(path/'test/test'), 

                         'has_cactus': y_test})
sub_df.head()
sub_df.to_csv('submission-v2.csv', index=False)