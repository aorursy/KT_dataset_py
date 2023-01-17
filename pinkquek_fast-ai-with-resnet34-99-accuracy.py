%reload_ext autoreload
%autoreload 2
%matplotlib inline
import torch
import glob
import os
import pathlib
import matplotlib.pyplot as plt
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH = "../input/10-monkey-species/"
sz=224
labels = np.array(pd.read_csv("../input/10-monkey-species/monkey_labels.txt", header=None, skiprows=1).iloc[:,2])
labels = [labels[i].strip() for i in range(len(labels))]
labels_name = ['Class: %d, %s'%(i,labels[i]) for i in range(10)]
def plots(ims, figsize=(12,6), rows=3, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows+1, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
imgs = []
for i in range(10):
    file = os.listdir(f'{PATH}training/training/n%d'%i)
    img = plt.imread(f'{PATH}training/training/n%d/{file[0]}'%i)
    imgs.append(img)

plots(imgs, titles=labels_name, rows=4, figsize=(16,15))
cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz), trn_name='training/training', val_name='validation/validation')
data.path = pathlib.Path('.') 
learn = ConvLearner.pretrained(arch, data, precompute=False)
learn.fit(0.01, 2)
log_preds = predict(learn.model,learn.data.val_dl)
preds = np.argmax(log_preds, axis=1)
def correct(is_correct): return np.where((preds == data.val_y)==is_correct)[0]

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = ['Prediction: %d, Truth: %d'%(preds[x],data.val_y[x]) for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))
plot_val_with_title(np.random.choice(correct(True),3,replace=False), "Correctly classified")
plot_val_with_title(correct(False), "All Incorrectly classified")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data.val_y, preds)
plot_confusion_matrix(cm, data.classes)
pltimgs = [plt.imread('../input/monkeys/'+name) for name in os.listdir('../input/monkeys/')]
plots(pltimgs, titles=os.listdir('../input/monkeys/'), rows=2, figsize=(16,15))
trn_tfms, val_tfms = tfms_from_model(arch,sz)
test_imgs = [val_tfms(open_image('../input/monkeys/'+name)) for name in os.listdir('../input/monkeys/')]
learn.precompute=False
test_pred = learn.predict_array(test_imgs)
test_pred = np.argmax(test_pred, axis=1)
test_pred
plots(pltimgs, titles=['Predict: '+labels[i] for i in test_pred], rows=2, figsize=(16,15))
