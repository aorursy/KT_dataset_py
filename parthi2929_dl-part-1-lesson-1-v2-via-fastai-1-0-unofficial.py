!pip freeze | grep fastai
from fastai import *
from fastai.vision import * 
from fastai.datasets import *  # this is needed for untar_data
from fastai.metrics import * # for accuracy
from fastai.vision.data import * 
import torch
torch.cuda.is_available()
torch.backends.cudnn.enabled
path = untar_data(URLs.DOGS)
sz=112  #size
path
import os
os.listdir(path)
from subprocess import check_output
print(check_output(["ls", "/tmp/.fastai/data/dogscats/train"]).decode("utf8"))
import numpy as np
fnames = np.array([f'train/cats/{f}' for f in sorted(os.listdir(f'{path}/train/cats'))] + [f'train/{f}' for f in sorted(os.listdir(f'{path}/train/dogs'))])
labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])
print(fnames)
import matplotlib.pyplot as plt 
img = plt.imread(f'{path}/{fnames[0]}')
plt.imshow(img);
img.shape
img[:4,:4]
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=sz, num_workers=1)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit(1)
# this gives prediction for test set  - afaik!
raw_preds,val_y,losses = learn.get_preds(with_loss=True) # predicting on test set
raw_preds.shape
# This is the label for a val data - afaik
val_y
# from here we know that 'dogs' is label 0 and 'cats' is label 1.
# Note this is opposite of original lesson 1 assignment
data.classes
# few predication values - probability in linear scale
raw_preds[:10]
preds = np.argmax(raw_preds, axis=1)  # from probabilities to 0 or 1
preds[:10]
probs = raw_preds[:,1]        # pr(dog)  # no exp operator caz tensor was in linear scale
np.around(probs[:10],3)       # printing in decimal as I could not tolerate exp format
def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == val_y)==is_correct)
def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
def load_img_id(ds, idx): return np.array(PIL.Image.open(ds.x.items[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.valid_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))
# 1. A few correct labels at random
plot_val_with_title(rand_by_correct(True), "Correctly classified")
# 2. A few incorrect labels at random
plot_val_with_title(rand_by_correct(False), "Incorrectly classified")
def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == val_y)==is_correct) & (val_y == y), mult)
plot_val_with_title(most_by_correct(0, True), "Most correct cats")
plot_val_with_title(most_by_correct(1, True), "Most correct dogs")
plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")
plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")
most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")
def simple_learner(): return Learner(data, simple_cnn((3,16,16,2)), metrics=[accuracy])
learn_s = simple_learner()
learn_s.lr_find(stop_div=False, num_it=250)
learn_s.recorder.plot_lr()
learn_s.recorder.plot()
import matplotlib.pyplot as plt
from random import randint

# fastai v0.7 definition for transforms_side_on not available in fastai v1.0
#transforms_basic    = [RandomRotate(10), RandomLighting(0.05, 0.05)]
#transforms_side_on  = transforms_basic + [RandomFlip()]

def get_augs():
    rand_deg = randint(0,10)
    rand_light = randint(0,5)/100
    tfms = get_transforms(xtra_tfms = [rotate(degrees=rand_deg),flip_lr()], max_zoom=1.1, max_lighting=rand_light)
    data_a = ImageDataBunch.from_folder(
        path, 
        ds_tfms = tfms,
        size=sz, 
        num_workers=1)     
    
    x,_ = next(iter(data_a.train_ds))
    return x
ims = np.stack([get_augs() for i in range(6)])
%matplotlib inline
# plt.imshow(image2np(ims[0].data))
# plt.show()
fig, axr = plt.subplots(2,3, figsize=(8,8))
k = 0
for i in range(2): # rows
    for j in range(3): # cols
        ims[k].show(ax=axr[i,j])
        k += 1
tfms = get_transforms(xtra_tfms = [rotate(degrees=10),flip_lr()], max_zoom=1.1, max_lighting=0.05)
data_a = ImageDataBunch.from_folder(
    path, 
    ds_tfms = tfms,
    size=sz, 
    num_workers=1)     
learn_a = create_cnn(data_a, models.resnet34, metrics=accuracy)
learn_a.fit(epochs=1, lr=1e-2)
learn_a.fit_one_cycle(3, 1e-2)
learn_a.recorder.plot_lr()
learn_a.save('224_lastlayer')
learn_a.load('224_lastlayer')
learn_a.unfreeze()
from fastai.callbacks import * 

def fit_sgd_warm(learn, n_cycles, lr, mom, cycle_len, cycle_mult):
    n = len(learn.data.train_dl)
    phases = [TrainingPhase(n * (cycle_len * cycle_mult**i), lr, mom, lr_anneal=annealing_cos) for i in range(n_cycles)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    if cycle_mult != 1:
        total_epochs = int(cycle_len * (1 - (cycle_mult)**n_cycles)/(1-cycle_mult)) 
    else: total_epochs = n_cycles * cycle_len
    learn.fit(total_epochs)
fit_sgd_warm(learn_a, 3, 1e-3, 0.9, 1, 2)
learn_a.recorder.plot_lr()
learn_a.save('224_all')
learn_a.load('224_all')
preds_a, y_a = learn_a.TTA()
def accuracy_np(preds, targs):
    preds = np.argmax(preds, 1)
    return (preds==targs).mean()

accuracy_np(preds_a.numpy(), y_a.numpy())
interp = ClassificationInterpretation.from_learner(learn_a)
interp.plot_confusion_matrix()
plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")
plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")
tfms = get_transforms(xtra_tfms = [rotate(degrees=10),flip_lr()], max_zoom=1.1, max_lighting=0.05)
data_z = ImageDataBunch.from_folder(
    path, 
    ds_tfms = tfms,
    size=sz, 
    num_workers=1)     
learn_z = create_cnn(data_z, models.resnet34, metrics=accuracy)
learn_z.fit(epochs=1, lr=1e-2)
def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
acts_z = np.array([1, 0, 0, 1])
preds_z = np.array([0.9, 0.1, 0.2, 0.8])
binary_loss(acts_z, preds_z)
!ls '/tmp/.fastai/data/dogscats/models'