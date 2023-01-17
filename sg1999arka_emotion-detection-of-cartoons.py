# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from fastai.metrics import accuracy
import pandas as pd

train = pd.read_csv("../input/Datasets/Train.csv")
train.head()
train['Emotion'].value_counts()
tfms = get_transforms(do_flip = True, flip_vert = False, max_rotate=10.0)
path = "../input/Datasets"
np.random.seed(0)

src = (ImageList.from_csv(path, 'Train.csv', folder='Train_Images')

      .split_by_rand_pct(0.2)

      .label_from_df())
data = (src.transform(tfms, size=128).databunch(bs=32).normalize(imagenet_stats))
data
data.show_batch(rows=3, figsize=(7,11))
arch = models.resnet152
from keras.utils.generic_utils import get_custom_objects

from keras import applications

from keras.layers import Activation

import keras

import tensorflow as tf
def swish(x,beta = 1):

    return x*sigmoid(beta*x)



get_custom_objects().update({'swish':Activation(swish)})
#opt_func:Callable='Adam'

learn = cnn_learner(data, arch, metrics=accuracy)

learn.model_dir='/kaggle/working/'
learn.lr_find(num_it=100)

learn.recorder.plot(suggestion=True)
#Fitting One Cycle Method in this Problem

lr = 1e-03

learn.fit_one_cycle(10,max_lr=lr)
learn.save('cartoon_emotions-1')
#Model Interpretation

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(9,figsize=(15,15))
test = ImageList.from_folder('../input/Datasets/Test_Images')

len(test)
test
learn.export(file = Path("/kaggle/working/export.pkl"))
deployed_path = "/kaggle/working/"
learn = load_learner(deployed_path, test = test)
preds, ids = learn.get_preds(ds_type=DatasetType.Test)
preds
learn.data.classes[:]
learn.data.classes[:]

len(preds)

preds[:10]
df = pd.DataFrame(preds, columns=learn.data.classes)

df.index+=1

df = df.assign(Label = df.values.argmax(axis=1))
df.head(10)
df = df.assign(image =  "fnames")

df = df.replace({'Label':{0:'Unknown', 1:'angry', 2:'happy', 3:'sad', 4:'surprised'}})

df = df.drop(['angry','happy','Unknown','sad','surprised'], axis=1)

df[:10]
df = df[['image', 'Label']]

df = df.rename({'image': 'Image'}, axis=1)

thresh = 0.30

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]

labelled_preds[:5]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]



fnames[:5]



suffix='.jpg'



fnames= [sub+suffix for sub in fnames]



learn.data.test_ds.items[:5]



df = pd.DataFrame({'Frame_ID':fnames, 'Emotion':labelled_preds})



df.head()
df.to_csv('submission.csv',index=False)
df.isnull().sum()