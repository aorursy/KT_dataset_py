%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
PATH = "../input/"
TMP_PATH = "/kaggle/working/tmp/"
MODEL_PATH = "/kaggle/working/models/"
sz = 224
arch = resnet34
bs = 64
m = arch(True)
m
m = nn.Sequential(*children(m)[:-2], 
                  nn.Conv2d(512, 2, 3, padding=1), 
                  nn.AdaptiveAvgPool2d(1), Flatten(), 
                  nn.LogSoftmax())
fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_names_and_array(
    path=PATH, 
    fnames=fnames, 
    y=labels, 
    classes=['dogs', 'cats'], 
    test_name='test', 
    tfms=tfms,
    bs=bs
)
learn = ConvLearner.from_model_data(m, data, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.freeze_to(-4)
m[-1].trainable
m[-4].trainable
learn.fit(0.01, 1)
learn.fit(0.01, 1, cycle_len=1)
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
x,y = next(iter(data.val_dl))
x,y = x[None,1], y[None,1]
vx = Variable(x.cuda(), requires_grad=True)
dx = data.val_ds.denorm(x)[0]
plt.imshow(dx);
sfs = [SaveFeatures(o) for o in [m[-7], m[-6], m[-5], m[-4]]]
%time py = m(Variable(x.cuda()))
for o in sfs: o.remove()
[o.features.size() for o in sfs]
py = np.exp(to_np(py)[0]); py
feat = np.maximum(0,to_np(sfs[3].features[0]))
feat.shape
f2=np.dot(np.rollaxis(feat,0,3), py)
f2-=f2.min()
f2/=f2.max()
f2
plt.imshow(dx)
plt.imshow(scale_min(f2, dx.shape[0]), alpha=0.5, cmap='hot');
learn.unfreeze()
learn.bn_freeze(True)
# 12 layer groups call for 12 lrs
lr=np.array([[1e-6]*4,[1e-4]*4,[1e-2]*4]).flatten()
learn.fit(lr, 2, cycle_len=1)
log_preds,y = learn.TTA()
preds = np.mean(np.exp(log_preds),0)
accuracy_np(preds,y)
learn.fit(lr, 2, cycle_len=1)
log_preds,y = learn.TTA()
preds = np.mean(np.exp(log_preds),0)
accuracy_np(preds,y)
