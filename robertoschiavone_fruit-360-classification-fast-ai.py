%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 16 # batch size

sz = 224 # image size
base_path = Path('../input/fruits-360_dataset/fruits-360/')

data = ImageDataBunch.from_folder(path=base_path, train='Training', valid='Test', size=sz, bs=bs, num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(6, 6))
print(data.classes)

len(data.classes), data.c
model_dir=Path('/tmp/models/')

learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=model_dir)

learn.model
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)
interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(32, 32), dpi=60)
interp.most_confused(min_val=2)