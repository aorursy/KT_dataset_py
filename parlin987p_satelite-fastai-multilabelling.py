%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai import *

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#donwload planet dataset

planet = untar_data(URLs.PLANET_SAMPLE)



# Any results you write to the current directory are saved as output.
df = pd.read_csv(planet/'labels.csv')

df.head()
# What are the different tags? 

df.pivot_table(index='tags', aggfunc=len).sort_values('image_name', ascending=False)
# GPU required

torch.cuda.is_available()
torch.backends.cudnn.enabled
# Fix to enable Resnet to live on Kaggle - creates a writable location for the models

cache_dir = os.path.expanduser(os.path.join('~', '.torch'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

   # print("directory created :" .cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)

  #  print("directory created :" . cache_dir)
#copying model to writable location

#cd /kaggle/working

!cp ../input/resnet152/resnet152.pth ~/.torch/models/resnet152-b121ed2d.pth
tfms = get_transforms(do_flip=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

np.random.seed(42)

data = ImageDataBunch.from_csv(planet, folder='train', size=128, suffix='.jpg', label_delim=' ', ds_tfms=tfms)  

data.normalize(imagenet_stats)
data.train_ds[0]
data.show_batch(rows=3, figsize=(10,9))
arch =  models.resnet152

# accuracy_thresh - selects the ones that are above a certain treshold 0.5 by default

acc_02 = partial(accuracy_thresh, thresh=0.2)  #partial function

f_score = partial (fbeta , thresh =0.2)

learn = create_cnn(data, arch, metrics =[acc_02,f_score])
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 0.015

learn.fit_one_cycle(5,slice(lr))
learn.save('stage-01-rn152')
learn.unfreeze() # unfreeze all layers



learn.lr_find() # find learning rate

learn.recorder.plot(suggestion=True) # plot learning rate
learn.fit_one_cycle(5,slice(1e-05,lr/5))
learn.save('stage-02-rn152')
# switch resolution to 256px0 0 create a transfer learning from what we created

data2 = ImageDataBunch.from_csv(planet, folder='train', size=256, suffix='.jpg', label_delim=' ', ds_tfms=tfms)  

data2.normalize(imagenet_stats)

#I am using the same learner with a new databunch

learn.data = data2

#training the last few layers

learn.freeze()
learn.lr_find() # find learning rate

learn.recorder.plot(suggestion=True) # plot learning rate
lr = 1e-2/2
learn.fit_one_cycle(5,slice(lr))
# saving, and unfreeze etc..