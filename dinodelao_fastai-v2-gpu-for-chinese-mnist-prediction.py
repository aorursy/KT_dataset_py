import fastai

from fastai.vision.all import *

from fastai.vision.widgets import *

import pandas as pd

import os
#imports files from kaggle

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#creates a path to the folder containing image files

path = Path("../input/chinese-mnist/data/data")



#makes .ls() format easier to read  

Path.BASE_PATH = path



#checks image files using .ls() method

path.ls()
df = pd.read_csv("../input/chinese-mnist/chinese_mnist.csv")

df.head()
df['fname'] = ("input_" + df['suite_id'].astype(str) 

               + "_" 

               + df['sample_id'].astype(str) 

               + "_" 

               + df['code'].astype(str) 

               + ".jpg")

df.head()
def get_x(r): return path/r['fname']



#.astype() and .split() method were added to contain each label

def get_y(r): return r['value'].astype(str).split(" ")
#creates a Datablock object

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),

                  splitter=RandomSplitter(seed=42),

                  get_x=get_x,

                  get_y=get_y,

                  item_tfms = RandomResizedCrop(128, min_scale=0.35))



# passes our dataframe into the dataloaders method of our DataBlock object

dls = dblock.dataloaders(df)
#displays number of batches for our training and validation sets

len(dls.train), len(dls.valid)
#displays a batch with images and labels

dls.show_batch(nrows=1, ncols=5)
#displays training Dataset

dls.train_ds
#displays validation Dataset

dls.valid_ds
#uses fastai's resnet18 model

learn = cnn_learner(dls, resnet18)



#creates a batch from our train dataset

x,y = to_cpu(dls.train.one_batch())



#generates predictions from our batch

batch = learn.model(x)
#analyzes batch

batch.shape
#we can index into our batch to return predictions for a single image

batch[0]
learn.loss_func
#defining our own sigmoid function

def sigmoid(x): return 1/(1+torch.exp(-x))



#defining our own BCELoss function  

def binary_cross_entropy(inputs, targets):

    inputs = inputs.sigmoid()

    return -torch.where(targets==1, inputs, 1-inputs).log().mean()
#creates a loss function

loss_func = nn.BCEWithLogitsLoss()



#passes our predictions and our labels into our loss function

loss = loss_func(batch, y)



#prints out or loss

loss
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):

    if sigmoid: inp = inp.sigmoid()

    return ((inp>thresh)==targ.bool()).float().mean()
learn = cnn_learner(dls, resnet18, metrics=partial(accuracy_multi, thresh=0.2))

learn.fine_tune(6)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(5, nrows=1)
interp.most_confused(5)
learn = cnn_learner(dls, resnet18, metrics=partial(accuracy_multi, thresh=0.2))

learn.lr_find()
learn = cnn_learner(dls, resnet18, metrics=partial(accuracy_multi, thresh=0.2))

learn.fine_tune(6, base_lr=1e-2)
learn = cnn_learner(dls, resnet18, metrics=partial(accuracy_multi, thresh=0.2))

learn.fit_one_cycle(3, base_lr=1e-2)
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(6, 1e-4)