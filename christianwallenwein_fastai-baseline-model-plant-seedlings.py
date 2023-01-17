%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import os

import pandas as pd
print(os.listdir("../input"))
train_dir = '../input/train/'

test_dir = '../input/test/'
print(os.listdir(train_dir)[:5])

print(os.listdir(test_dir)[:5])
tfms = get_transforms()

data = ImageDataBunch.from_folder(

    path = train_dir,

    test="../test",

    valid_pct = 0.2,

    bs = 16,

    size = 336,

    ds_tfms = tfms,

    num_workers = 0

).normalize(imagenet_stats)

data

print(data.classes)

data.show_batch()
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_multi_top_losses(3, figsize=(6,6))
interp.plot_confusion_matrix()
interp.most_confused(min_val=5)
class_score, y = learn.get_preds(DatasetType.Test)

class_score = np.argmax(class_score, axis=1)
predicted_classes = [data.classes[i] for i in class_score]

predicted_classes[:10]
submission  = pd.DataFrame({

    "file": os.listdir(test_dir),

    "species": predicted_classes

})

submission.to_csv("submission.csv", index=False)

submission[:10]