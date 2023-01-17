import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Following the notes from fast.aitutorial docs - https://docs.fast.ai/tutorial.data.html
# Adding Resnet 18 dataset

#Learnt a lot from these two.. https://www.kaggle.com/hortonhearsafoo/fast-ai-lesson-2
#From: https://nbviewer.jupyter.org/github/arunoda/fastai-courses/blob/master/dl1/lesson3-planet.ipynb#Test-Dataset

# First of all, we need to find some metrics 
# Basically these are just for printing only.
# Since this is a multi classification problem, we need to use a threshold for the 
# accuracy.
def p_accuracy(pred, act, **kwargs):
    return accuracy_thresh(pred, act, thresh=0.2, **kwargs)
#This kaggle competition uses f2 score for the final eval. So we should use that as well.
def f2_score(pred, act, **kwargs):
    return fbeta(pred, act, beta=2, thresh=0.2, **kwargs)

%reload_ext autoreload
%autoreload 2
%matplotlib inline
# This file contains all the main external libs we'll use - fastai v1
from fastai import *
from fastai.vision import *
debug = 0
PATH = "/kaggle/input/planet-understanding-the-amazon-from-space/"
# 32 when testing variable building to 256 when for real
if debug:
    sz=32 
    print("In low res debug mode - quick but not accurate at all")
else:
    sz=256
    print("In high res mode - slow, looking for that final result")
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
arch = 'resnet18' #just for naming the submission file
comp_name = "planet"
!ls {PATH} # directory for training and test files
label_df = pd.read_csv(f'{PATH}train_v2.csv')
#what does the csv file look like id is the file name (minus .jpg), breed is the classification
label_df.head()
# What are the different tags? 
label_df.pivot_table(index='tags', aggfunc=len).sort_values('image_name', ascending=False)
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
shutil.copy("/kaggle/input/resnet18/resnet18.pth", "/tmp/.torch/models/resnet18-5c106cde.pth")
tfms = get_transforms(do_flip=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

def get_data(sz):
    tfms = get_transforms(do_flip=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    
    data = (
        ImageItemList
            .from_csv(PATH, 'train_v2.csv', folder="train-jpg", suffix=".jpg")
            .random_split_by_pct(0.2)
            .label_from_df(sep=' ')
            .transform(tfms, size=sz)
            .add_test_folder('test-jpg-v2')
            .databunch(num_workers=0)
            .normalize(imagenet_stats)
    )
    return data
#crashes with too many workers (remove numworkers while testing with a small image size ~ 64), keep for the bigger images for accuracy

#Cause learning on 64x64
sz=64
data = get_data(sz)
len(data.classes), data.classes
data.show_batch(rows=3, figsize=(10,12))
img = plt.imread(f'{PATH}train-jpg/{label_df.iloc[0,0]}.jpg')
plt.imshow(img);
# all images are 256 x 256
img.size
learn = create_cnn(data, models.resnet18, model_dir=MODEL_PATH, metrics=[p_accuracy])
learn.lr_find()
learn.recorder.plot()
lr = 1e-1
learn.fit_one_cycle(5, slice(lr))
learn.metrics = [p_accuracy, f2_score]
learn.unfreeze() 
learn.lr_find()
learn.recorder.plot()
if debug:
    learn.fit_one_cycle(1, max_lr=(1e-6, 1e-5, 1e-4))
else:
    learn.fit_one_cycle(3, max_lr=(1e-6, 1e-5, 1e-4))
sz=128
learn.data = get_data(sz)
learn.freeze()
learn.lr_find()
learn.recorder.plot()
if debug:
    learn.fit_one_cycle(1, slice(1e-5, 1e-3))
else: 
    learn.fit_one_cycle(3, slice(1e-5, 1e-3))
sz=256
learn.data = get_data(sz)
learn.freeze()
learn.lr_find()
learn.recorder.plot()
if debug:
    learn.fit_one_cycle(1, slice((1e-2)/2, (1e-1)/2))
else:
    learn.fit_one_cycle(3, slice((1e-2)/2, (1e-1)/2))
learn.unfreeze();
learn.lr_find()
learn.recorder.plot()
if debug:
    learn.fit_one_cycle(1,  slice(1e-6, 1e-5))
else:
    learn.fit_one_cycle(5,  slice(1e-6, 1e-5))
learn.show_results(rows=3, figsize=(12,15))
# from https://nbviewer.jupyter.org/github/arunoda/fastai-courses/blob/master/dl1/lesson3-planet.ipynb#Test-Dataset
def get_tags(pred, thresh):
    classes = ""
    best_guess = ""
    tags = 0
    high_val = 0
    if debug:
        thresh = 0.2
        print(f"Debug - using low threshold {thresh}")
    for idx, val in enumerate(pred):
        if val > thresh:
            classes = f'{classes} {learn.data.classes[idx]}'
            tags = tags+1
        if val > high_val:
            high_val = val
            best_guess = f'{learn.data.classes[idx]}'
    if tags == 0:
        classes = best_guess
    return classes.strip()
def predict(idx):
    pred_vals = predictions[0][idx]
    tags = get_tags(pred_vals, 0.2)
    print(tags)
    img = learn.data.test_ds[idx][0]
    return img
def get_row(idx):
    pred = predictions[0][idx]
    tags = get_tags(pred, 0.2)
    image_path = learn.data.test_ds.x.items[idx]
    image_name = re.search(r'([^/]+)$', f'{image_path}')[0].replace('.jpg', '') 
    return image_name, tags
len(learn.data.test_ds)
predictions = learn.get_preds(ds_type=DatasetType.Test)
predict(0)
df = pd.DataFrame(columns=['image_name', 'tags'])
for idx in range(len(predictions[0])):
    if idx % 5000 == 0:
        print(f"Completed: {idx}")
        
    image_name, tags = get_row(idx)
    df.loc[idx] = [image_name, tags]
df.head()
dt = date.today()
date_str = dt.isoformat()
submission_path = f'submission-256-{date_str}-size.csv'
df.to_csv(submission_path, index=False)
!head {submission_path}