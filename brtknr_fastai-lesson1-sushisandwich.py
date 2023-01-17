# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
!mkdir -p /tmp/.torch/models
!mkdir -p data
!ln -s /kaggle/input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
!ln -s /kaggle/input/sushisandwich/train data/train
!ln -s /kaggle/input/sushisandwich/valid data/valid
# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH='data/'
sz=224
!ls {PATH}
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir(PATH)
LABELS = os.listdir(f'{PATH}/valid')
LABELS
files = os.listdir(f'{PATH}/valid/{LABELS[0]}')[:5]
files
img = plt.imread(f'{PATH}/valid/{LABELS[0]}/{files[0]}')
plt.imshow(img);
img.shape
img[:4,:4]
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 2)
# This is the label for a val data
data.val_y
# from here we know that 'cats' is label 0 and 'dogs' is label 1.
data.classes
# this gives prediction for validation set. Predictions are in log scale
log_preds = learn.predict()
log_preds.shape
log_preds[:10]
preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1
probs = np.exp(log_preds[:,1])        # pr(dog)
def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)
def plot_val_with_title(idxs, title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probs)
def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
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
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)
plot_val_with_title(most_by_correct(0, True), "Most correct sandwich")
plot_val_with_title(most_by_correct(1, True), "Most correct sushi")
plot_val_with_title(most_by_correct(0, False), "Most incorrect sandwich")
plot_val_with_title(most_by_correct(1, False), "Most incorrect sushi")
most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")
learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf=learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]
ims = np.stack([get_augs() for i in range(6)])
plots(ims, rows=2)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(1e-2, 3)
phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-3,1e-2), lr_decay=DecayType.LINEAR, 
                          momentum=(0.95,0.85), momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2,1e-3), lr_decay=DecayType.LINEAR, 
                          momentum=(0.85,0.95), momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=1, opt_fn=optim.Adam, lr=1e-3, lr_decay=DecayType.COSINE, momentum=0.9)]
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit_opt_sched(phases)
learn.sched.plot_lr(show_text=True)
learn.precompute=False
learn.fit(1e-2, 3, cycle_len=1)
learn.sched.plot_lr()
learn.unfreeze()
lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()
learn.save('224_all')
learn.load('224_all')
learn.freeze()
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)
plot_val_with_title(most_by_correct(0, False), f"Most incorrect {LABELS[0]}")
plot_val_with_title(most_by_correct(1, False), f"Most incorrect {LABELS[1]}")
tfms = tfms_from_model(resnet34, sz)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(resnet34, data, precompute=True)
learn.fit(1e-2, 1)
def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
acts = np.array([1, 0, 0, 1])
preds = np.array([0.9, 0.1, 0.2, 0.8])
binary_loss(acts, preds)
