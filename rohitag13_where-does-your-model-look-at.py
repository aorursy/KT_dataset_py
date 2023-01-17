import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from pathlib import Path
from glob import glob
from PIL import Image
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
PATH = Path('../input')
trn_df = pd.read_csv(PATH/f'train_labels.csv')
trn_df.head()
trn_df.columns = [0,1]

trn_df.to_csv('../working/trn.csv',index=False)

path = Path('../input/train')

tfms = []
src = ImageItemList.from_df(trn_df,'../input/train',suffix='.tif')
src = src.random_split_by_pct()
src = src.label_from_df()
src = src.transform(tfms)
src = src.databunch(path='../input/train').normalize(imagenet_stats)
x,y = src.one_batch(DatasetType.Train,True,True)
grab_idx(x,63).size()
src.show_batch(3)
MODEL_PATH = "/tmp/model/"
gc.collect()
learn = create_cnn(src, models.resnet34, metrics=error_rate, bn_final=True,model_dir=MODEL_PATH)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-3), pct_start=0.8)
m = learn.model.eval();
## was trying to make code modular : TODO
class Heatmap:
    def __init__(self, data, model,xs, ys, idxs):
        self.data, self.m = data, model
        # self.ds = data.train_ds if mode == 'train' else data.valid_ds
        self.xs, self.ys ,self.idxs = xs, ys, idxs
        # self.xb,_ = self.data.one_item(self.xs)

    def get_hm(self,idx): # this will be called by plot_* func
        
        xb_im = Image(self.data.denorm(self.xb)[0]) # this will the image to display
        self.xb = self.xb.cuda() # pushed the xb to GPU
        hook_a, hook_g = self.hooked_backward()
        acts  = hook_a.stored[0].cpu()
        avg_acts = acts.mean(0)
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        hm = (acts * grad_chan[...,None,None]).mean(0)
        return xb_im, y, hm
    @classmethod
    def plot_rand(cls, data, model, mode='train'):
        pass
    @classmethod
    def plot_pos(cls, data, model, count = 4, mode='train'):
        dtype = DatasetType.Train if mode == 'train' else DatasetType.Valid
        xs, ys = data.one_batch(dtype, True, True)
        idxs = np.argwhere(y)[0][:count]
        return cls(data, model, xs, ys, idxs)
    @classmethod
    def plot_neg(cls, data, model, mode='train'):
        pass
    
    def hooked_backward(self):
        with hook_output(self.m[0]) as hook_a: 
            with hook_output(self.m[0], grad=True) as hook_g:
                preds = self.m(self.xb)
                preds[0,int(self.y)].backward()
        return hook_a,hook_g
def show_heatmap():
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,96,96,0),
              interpolation='bilinear', cmap='magma');
def get_xb_im(idx):
    x,y = src.valid_ds[idx]
    xb,_ = src.one_item(x)
    xb_im = Image(src.denorm(xb)[0])
    return xb_im,y
def get_hm(idx):
    x,y = src.train_ds[idx]
    xb,_ = src.one_item(x)
    xb = xb.cuda()
    hook_a,hook_g = hooked_backward(y,m,xb)
    acts  = hook_a.stored[0].cpu()
    avg_acts = acts.mean(0)
    grad = hook_g.stored[0][0].cpu()
    grad_chan = grad.mean(1).mean(1)
    hm = (acts*grad_chan[...,None,None]).mean(0)
    return hm
def hooked_backward(cat,m,xb):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g
def plot(r,c,cmap='magma',figsize=(10,10)):
    axes = plt.subplots(r, c, figsize=figsize)[1]
    for i in range(r):
        for j in range(c): 
            INDEX = np.random.randint(0,1000)
            hm = get_hm(INDEX)
            xb_im,y = get_xb_im(INDEX)
            xb_im.show(axes[i][j])
            axes[i][j].imshow(hm, alpha=0.6, extent=(0,96,96,0),
                      interpolation='bilinear', cmap=cmap);
            axes[i][j].set_title(y)
            plt.axis('off')
            plt.tight_layout()
plot(4,4,figsize=(10,10))
plot(4,4,'PuOr',figsize=(10,10))
plot(4,4,'plasma')
plot(4,4,'inferno')
plot(4,4,'cividis')
plot(4,4,'cool')
plot(4,4,'hsv')
plot(4,4,'hsv')
plot(4,4,'tab20b')
plot(4,4,'flag')
plot(4,4,'spring')
