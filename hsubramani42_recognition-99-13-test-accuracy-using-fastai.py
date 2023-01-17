from fastai import *
from fastai.vision import *
path='../input/flowers-recognition'
data=ImageDataBunch.from_folder(path,valid_pct=0.20,ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)
data
data.show_batch(rows=3,fig=(5,5))
model=cnn_learner(data,models.densenet201,metrics=accuracy)
model.summary()
model.fit(5)
model.recorder.plot_losses()
model.recorder.plot_metrics()
model.recorder.plot_lr()
interp=ClassificationInterpretation.from_learner(model)
interp.plot_confusion_matrix(title='Confusion matrix')