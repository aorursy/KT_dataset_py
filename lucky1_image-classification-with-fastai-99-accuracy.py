%reload_ext autoreload

%autoreload 2

%matplotlib inline
import sys

import fastai



print("Python Version: " + sys.version) # python version

print("FastAI Version: " + fastai.__version__) # fastai version



from fastai import *

from fastai.vision import *
dataset_dir = "../input/10-monkey-species/"
data = ImageDataBunch.from_folder(dataset_dir, train= "training", valid="validation", 

                                  ds_tfms=get_transforms(), size=224, bs=64)



data.normalize(imagenet_stats)
print(data.classes)

len(data.classes)
data.show_batch(rows=3, figsize=(7,6))
import os

path = '/root/.cache/torch/checkpoints/'

os.makedirs(path, exist_ok=True)    

    

!cp ../input/resnet34/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model_dir = "../../output"
learn.fit_one_cycle(2)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()  
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=1)