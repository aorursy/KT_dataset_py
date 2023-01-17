!pip install git+git://github.com/fastai/fastai.git
import numpy as np 
import pandas as pd 
import fastai
fastai.__version__
# import vision submodule from fastai
from fastai.vision.all import *
# path to directory of images
path = '/kaggle/input/crowd-counting/frames/frames/'
# sample of paths to input images
Path(path).ls()[:3]
df = pd.read_csv('/kaggle/input/crowd-counting/labels.csv')
df.head()
df['id'] = df.id.apply(lambda x: '/kaggle/input/crowd-counting/frames/frames/seq_{:06d}.jpg'.format(x))
df.columns = ['x','y']
df.head()
def get_x(df): return df['x'] # path to images
def get_y(df): return df['y'] # labels

d = DataBlock(blocks=(ImageBlock, RegressionBlock),    # types of input and output
         get_x = get_x,                                # function to get path to images
         get_y = get_y,                                # function fo get labels
         splitter=RandomSplitter(),                    # random splitter (20% for validation)
         item_tfms=Resize(224, ResizeMethod.Squish),)  # resize and squish images 
dl = d.dataloaders(df)
len(dl.train_ds), len(dl.valid_ds)
dl.show_batch(figsize=(12,12))
learn = cnn_learner(dl, resnet18, metrics=mae)
learn.fine_tune(8, 5e-2)
learn.show_results(figsize=(12,12))