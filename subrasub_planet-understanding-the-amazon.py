%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np 
import pandas as pd
from fastai.vision import *
from fastai import *
path = untar_data(URLs.PLANET_SAMPLE)

df = pd.read_csv(path/'labels.csv')
df.head()
tfms = get_transforms(do_flip=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)
data = ImageDataBunch.from_csv(path, folder='train', size=256, suffix='.jpg', label_delim=' ', ds_tfms=tfms)  
data.normalize(imagenet_stats)
data.train_ds[0]
data.valid_ds.classes
data.show_batch(rows=3, figsize=(10,9))
arch =  models.resnet50
# accuracy_thresh - selects the ones that are above a certain treshold 0.5 by default
acc_02 = partial(accuracy_thresh, thresh=0.2)  #partial function
f_score = partial (fbeta , thresh =0.2)
learn = create_cnn(data, arch, metrics =[acc_02,f_score])
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = 0.01
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50') 
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1e-4, lr/5))
learn.save('stage-2-rn50')
