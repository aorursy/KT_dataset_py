# enable IE to call matlplotlib
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#??fastai.imports
# This file contains all the main external libs we'll use
# Here we import other libraries except fastai (or other deep learning tools)
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH = '../input/'
sz = 224
arch = resnet34
bs = 128
!ls {PATH}
print('CUDA is available?', torch.cuda.is_available())
print('pytorch version:', torch.__version__)
torch.backends.cudnn.enabled
files = os.listdir(f'{PATH}train')[:5]
files
img = plt.imread(f'{PATH}train/{files[0]}')
plt.imshow(img);
img.shape
img[:4,:4]
label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n)
val_idxs
# make soft link for dataset only if using Kaggle Kernel
!ln -s {PATH}train 
!ln -s {PATH}test
!ls
PATH = '.'
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', label_csv, bs=bs, tfms=tfms,  
                            suffix='.jpg', val_idxs=val_idxs, num_workers=4)
#??ConvLearner.pretrained
learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)
lrf = learn.lr_find()
learn.sched.plot()
# learning rate corresponding to the lowset point location, then div 10 
#learn.fit(0.5, 3)
# different batch size corresponding to different best learning rate
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
def get_augs():
    data = ImageClassifierData.from_csv(PATH, 'train', label_csv, bs=bs, tfms=tfms,
                    val_idxs=val_idxs, suffix='.jpg', num_workers=4)
    x, _ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]
ims = np.stack([get_augs() for i in range(6)])
plots(ims, rows=2)
# create new data object including this augmentation
data = ImageClassifierData.from_csv(PATH, 'train', label_csv, bs=bs, tfms=tfms, 
                        val_idxs=val_idxs, suffix='.jpg', num_workers=4)
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.5, 1)
# As we want to use data augmentation, should find a new best learning rate
lrf = learn.lr_find()
learn.sched.plot()
learn.precompute = False
learn.fit(0.1, 3, cycle_len=1)
learn.sched.plot_lr()
learn.save('224_lastlayer')
learn.load('224_lastlayer')
!ls 'models'
learn.unfreeze()
# After unfreeze the former layers, we should find learning rate again
learn.lr_find()
learn.sched.plot()
#lr = np.array([5e-5, 5e-4, 5e-3])
lr = np.array([1e-4, 1e-3, 1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()
learn.save('224_all')
learn.load('224_all')
# Using FileLink can generate Link to download files, like, trianed model
#FileLink('./models/224_all.h5')
log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)

plot_confusion_matrix(cm, data.classes)
!rm -rf train
!rm -rf test
!rm -rf tmp/
!ls
