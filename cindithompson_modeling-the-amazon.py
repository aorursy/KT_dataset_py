import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))
# This import contain all the main external libs we'll use

from fastai.vision import *
debug = 1

PATH = "/kaggle/input/planet-understanding-the-amazon-from-space/"

# 32 when testing variable building to 256 when for real

if debug:

    sz=32 

    print("In low res debug mode - quick but not accurate at all")

else:

    sz=256

    print("In high res mode - slow, looking for that final result")

MODEL_PATH = "/tmp/model/"
!ls {PATH} # directory for training and test files
# GPU required

torch.cuda.is_available()
torch.backends.cudnn.enabled
np.random.seed(42)  # for reproducibility

rescaled_dim = 20

bs = 64

num_workers = 0  
labels_df = pd.read_csv(f'{PATH}train_v2.csv')

num_exs = len(labels_df)

ntrain = int(num_exs * .6)

nval = int((num_exs-ntrain)/2)
src = (ImageItemList.from_csv(PATH, 'train_v2.csv', folder="train-jpg", suffix=".jpg")

       .split_by_idxs(list(range(ntrain)),valid_idx=list(range(ntrain,ntrain+nval)))  # get the same training data as baseline

       .label_from_df(sep=' ')     # one-hot encoding

      )
data = (src.transform(tfms=None, size=rescaled_dim)  # resize

        .databunch(bs=bs, num_workers=num_workers) # format needed for training

        .normalize(imagenet_stats))  # like sklearn.preprocessing.scale, with some twists
print(len(data.train_ds))

print(len(data.valid_ds))
data.show_batch(rows=3, figsize=(10,12))
arch = models.resnet50
#This kaggle competition uses f_2 score for the final eval. 

# So we will use that as well.

def f2_score(pred, act, **kwargs):

    return fbeta(pred, act, beta=2, thresh=0.2, **kwargs)
learn = create_cnn(data, arch, metrics=[f2_score], model_dir='/tmp/models')

learn.fit(1)