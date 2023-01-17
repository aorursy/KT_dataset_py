# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import imageio



%matplotlib inline
import fastai

from fastai.vision import *

from fastai.metrics import error_rate

print(fastai.__version__) # Checking version
verbose = 0

train = True  ## DO NOT CHANGE 'train' to False - See NOTE below ##

# Idea was to mark 'train' as False before submission but this deoes not help to speed up submission.

# Hence not using 'train' anymore, but not deleting for now (set always to True)
root = '/kaggle/input/'

parent = "Kannada-MNIST/"



root_plus_parent = root + parent



if train:

    if verbose >= 1:

        print(root_plus_parent)



    if verbose >= 1:

        for dirname, _, filenames in os.walk(root):

            for filename in filenames:

                print(os.path.join(dirname, filename))
if train:

    df_train = pd.read_csv(root_plus_parent + "train.csv")

    

if train & verbose >= 1:

    print(df_train.head())
df_test = pd.read_csv(root_plus_parent + "test.csv")

if train & verbose >= 1:

    print(df_test.head())
if train & verbose >= 1:

    print(df_train.shape, df_test.shape)
if train:

    train_x = df_train.iloc[:,1:].values.reshape(-1,28,28)

    

if train & verbose >= 1:

    print(train_x.shape)
if train:

    train_y = df_train['label'].values



if train & verbose >= 1:

    print(train_y.shape)
test_x = df_test.iloc[:,1:].values.reshape(-1,28,28)

if train & verbose >= 1:

    print(test_x.shape)
if train & verbose >= 1:

    plt.imshow(train_x[0,:], cmap='gray')
# This function has been taken from another kernel: https://www.kaggle.com/demonplus/kannada-mnist-with-fast-ai-and-resnet

def save_imgs(path:Path, data, labels=[]):

    path.mkdir(parents=True,exist_ok=True)

    for label in np.unique(labels):

        (path/str(label)).mkdir(parents=True,exist_ok=True)

    for i in range(len(data)):

        if(len(labels)!=0):

            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )

        else:

            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )  # For test data which does not have a label
path = "../input/data"

train_path = Path('../input/data/train')

test_path = Path('../input/data/test')
if train:

    save_imgs(train_path,train_x,train_y)



if train & verbose >= 1:

    print((train_path).ls())
save_imgs(test_path,test_x)



if train & verbose >= 1:

    print((test_path).ls())
if train:

    np.random.seed(42)  # Make sure your validation set is reproducible

    # tfms = get_transforms(do_flip=False)  # We dont want to flip the images since numbers are not written flipped

    tfms = get_transforms(do_flip=False, max_rotate=30, max_zoom=1.2, max_lighting=0.4,

                          max_warp=0.3)  # We dont want to flip the images since numbers are not written flipped + more aggressive augmentation



    data = ImageDataBunch.from_folder(path=path,  

                                      valid_pct=0.2,

                                      ds_tfms=tfms,

                                      train='train',

                                      test='test',

                                      size=28,

                                      bs=64).normalize(imagenet_stats)
if train & verbose >= 1:

    print(len(data.train_ds))

    print(len(data.valid_ds))

    print(len(data.test_ds))
if train & verbose >= 1:

    data.show_batch(rows=6, figsize=(12,12))



# Subtle differences between the following which might be hard to detect

# - 9 and 6

# - 1 and 0

# - 7 and 3
if train & verbose >= 1:

    print(data.classes)
# Kaggle comes with internet off. So have to copy over model to the location where fastai would have downloaded it.

# https://forums.fast.ai/t/how-can-i-load-a-pretrained-model-on-kaggle-using-fastai/13941/23



!mkdir -p /tmp/.cache/torch/checkpoints

!cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth
if train:

    learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])
if train & verbose >= 1:

    print(learn.model)
if train:

    learn.fit_one_cycle(4)
if train:

    learn.recorder.plot_losses()
if train:

    learn.save('resnet50-stage-1')  # Saves it in the directory where the images exist (under a folder called 'models' in there)
if train:

    learn.lr_find()

    learn.recorder.plot()
if train:

    learn.unfreeze()  # Train all layers

    learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-3))  # As per lr_finder (initial layers trained on smaller learning rate)
if train:

    learn.recorder.plot_losses()
if train:

    learn.save('resnet50-stage-2a')
if train:

    learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-3))  
if train:

    learn.recorder.plot_losses()
if train:

    learn.save('resnet50-stage-2b')
if train:

    learn.export(file = Path("/kaggle/working/export.pkl"))  # Not needed unless you want to download the model and then upload as a dataset
if train:

    interp = ClassificationInterpretation.from_learner(learn)
if train:

    interp.plot_confusion_matrix()
if train:

    print(interp.most_confused())
if train:

    interp.plot_top_losses(9, figsize=(10,10))
# https://docs.fast.ai/tutorial.inference.html#A-classification-problem

if train:

    deployed_path = "/kaggle/working/"

else:

    deployed_path = "../input/kannada-mnist-resnet50/"  # If using a deployed model (not using right now)

    

print(deployed_path)
learn = load_learner(deployed_path, test=ImageList.from_folder(test_path))
# Adapted from another kernel: https://www.kaggle.com/demonplus/kannada-mnist-with-fast-ai-and-resnet

test_preds_probs, _ = learn.get_preds(DatasetType.Test)

test_preds = torch.argmax(test_preds_probs, dim=1)

if verbose >= 1:

    print(test_preds_probs)
num = len(learn.data.test_ds)

indexes = {}



for i in range(num):

    filename = str(learn.data.test_ds.items[i]).split('/')[-1]

    filename = filename[:-4] # get rid of .jpg

    indexes[(int)(filename)] = i
submission = pd.DataFrame({'id': range(0, num), 'label': [test_preds[indexes[x]].item() for x in range(0, num)] })

print(submission)
submission.to_csv(path_or_buf ="submission.csv", index=False)