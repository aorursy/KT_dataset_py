! mkdir ../data

! mkdir ../data/train

! mkdir ../data/validation

! cp -r ../input/training/training/* ../data/train/

! cp -r ../input/validation/validation/* ../data/validation/
from fastai import *

from fastai.vision import *

from pathlib import Path
data = ImageDataBunch.from_folder(path=Path('../data').resolve(), train='train', valid='validation', dl_tfms=get_transforms(), num_workers=0, bs=64, size=224).normalize(imagenet_stats)
# import pandas as pd



# labels_df = pd.read_csv('../input/monkey_labels.txt', delimiter=' *, *', engine='python')

# labels = dict(zip(labels_df['Label'].tolist(), labels_df['Common Name'].tolist()))
# data = ImageImageList.from_folder(path=Path('../data').resolve()).split_by_folder(train='train', valid='validation').label_from_func(func=lambda x: labels[str(x.parts[-2])]).transform(get_transforms(), size=224).databunch(num_workers=0, bs=64).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(5,5))
learn_34 = create_cnn(data, models.resnet34, metrics=[accuracy, error_rate])

learn_34.fit_one_cycle(3)
interp_34 = ClassificationInterpretation.from_learner(learn_34)

interp_34.plot_top_losses(9, figsize=(15,11))
interp_34.plot_confusion_matrix()
learn_34.lr_find()

learn_34.recorder.plot()
learn_34.fit_one_cycle(1, slice(2e-4, 9e-2))
learn_50 = create_cnn(data, models.resnet50, metrics=[accuracy, error_rate])

learn_50.fit_one_cycle(3)
interp_50 = ClassificationInterpretation.from_learner(learn_50)

interp_50.plot_confusion_matrix()
interp_50.plot_top_losses(4, figsize=(15,11))
learn_50.lr_find()

learn_50.recorder.plot()
learn_50.fit_one_cycle(2, slice(8e-06, 1.2e-06))