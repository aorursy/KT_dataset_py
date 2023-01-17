# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#Lesson 1
import os
import matplotlib.pyplot as plt
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
print(torch.cuda.is_available())
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=224
train_cats = os.listdir(f'{PATH}train/cats')
train_dogs = os.listdir(f'{PATH}train/dogs')
valid_cats = os.listdir(f'{PATH}valid/cats')
valid_dogs = os.listdir(f'{PATH}valid/dogs')
print("train_cats={}".format(len(train_cats)))
print("train_dogs={}".format(len(train_dogs)))
print("valid_cats={}".format(len(valid_cats)))
print("valid_dogs={}".format(len(valid_dogs)))
img_cat = plt.imread(f'{PATH}train/cats/{train_cats[0]}')
plt.imshow(img_cat);
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
img = PIL.Image.open(PATH + data.trn_ds.fnames[0])
plt.imshow(img)
size_d = {k: PIL.Image.open(PATH + k).size for k in data.trn_ds.fnames}
row_sz, col_sz = list(zip(*size_d.values()))
plt.hist(row_sz)
#data.val_y,data.classes
plt.hist(col_sz)
learn = ConvLearner.pretrained(arch, data, precompute=True,tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.01, 3)
log_preds = learn.predict()
log_preds.shape
preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1
probs = np.exp(log_preds[:,1]) 
def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], min(len(preds), 4), replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)
def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))
def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))
def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])       
def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]
def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)
plot_val_with_title(rand_by_correct(True),"Correctly Classified")
plot_val_with_title(rand_by_correct(False),"Incorrectly Classified")
plot_val_with_title(most_by_correct(0, True), "Most correct cats")
plot_val_with_title(most_by_correct(1, True), "Most correct dogs")
plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")
plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")
most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
lrf=learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
#Lesson 2
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(
    path=PATH, 
    tfms=tfms
)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.01,2)
learn.precompute=False
learn.fit(1e-2, 3, cycle_len=1)
learn.unfreeze()
lr = 0.01
lrn = np.array([lr/100,lr/10,lr])
learn.fit(lrn,3,cycle_len=1, cycle_mult=2)
log_preds,y = learn.TTA()
probs=np.mean(np.exp(log_preds),0)
accuracy(probs,y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)
