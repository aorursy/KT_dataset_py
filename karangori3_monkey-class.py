import warnings
warnings.filterwarnings('ignore')

#setting up our enviroment
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#importing libraries
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
x  = '../input/10-monkey-species/training/training'
path = Path(x)
path.ls()
np.random.seed(40)
data = ImageDataBunch.from_folder(path, train = '.', valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=224,
                                  num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6),recompute_scale_factor=True)
data
print(data.classes)
len(data.classes)
learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
lr1 = 1e-3
lr2 = 1e-1
learn.fit_one_cycle(5,slice(lr1,lr2))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
#with open('../input/10-monkey-species/monkey_labels.txt') as file:
common_name = pd.read_csv('../input/10-monkey-species/monkey_labels.txt')
common_name.rename(columns = {" Common Name                   ":"Common Name"}, inplace = True)
common_name['Label'] = common_name['Label'].str.strip()
common_name
img = open_image('../input/10-monkey-species/validation/validation/n0/n000.jpg')
label = str(learn.predict(img)[0])
print("Predicted label : {}".format(label))
img
learn.export(file = Path("/kaggle/working/export.pkl"))
learn.model_dir = "/kaggle/working"
learn.save("stage-1",return_path=True)