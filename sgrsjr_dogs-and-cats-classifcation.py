import os, glob
from sklearn.model_selection import train_test_split
prefix_test = '../input/dogscats/test_set/'
prefix_train = '../input/dogscats/training_set/'
PATH = 'data/dogscats/'
cwd = os.getcwd() + '/'
# making required folders
os.makedirs(PATH + 'train', exist_ok=True)
os.makedirs(PATH + 'test', exist_ok=True)
os.makedirs(PATH + 'valid', exist_ok=True)
os.makedirs(PATH + 'models', exist_ok=True)
# creating symlinks for the data in the working directory
for label in ['cat', 'dog']:
    list = [os.path.basename(file) for file in glob.glob(prefix_train + label+'s/' + label + '.*')]
    
    train_set, valid_set = train_test_split(list, test_size=0.2, random_state=42)
    test_set = [os.path.basename(file) for file in glob.glob(prefix_test + label+'s/' + label + '.*')]
    
    print(label+'s')
    print('Training Images:', len(train_set))
    print('Validation Images:', len(valid_set))
    print('Test Images :', len(test_set), end='\n\n')
    
    for dest in ['train', 'valid', 'test']:
        dest_prefix = PATH + dest+'/'
        os.makedirs(dest_prefix + label+'s/', exist_ok=True)
        for file in eval(dest + '_set'):
            if dest == 'test': os.symlink(cwd + prefix_test + label+'s/' + file, cwd + dest_prefix + label+'s/' + file)
            else: os.symlink(cwd + prefix_train + label+'s/' + file, cwd + dest_prefix + label+'s/' + file)
!ls {PATH}
print(os.listdir(PATH + 'train'))
print(os.listdir(PATH + 'test'))
print(os.listdir(PATH + 'valid'))
import matplotlib.pyplot as plt
%matplotlib inline
files = os.listdir(f'{PATH}valid/cats')
img = plt.imread(f'{PATH}valid/cats/{files[0]}')
plt.imshow(img);
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
print('GPU available:', torch.cuda.is_available())
print('CuDNN available:', torch.backends.cudnn.enabled)
arch = resnet34
sz = 224
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz), test_name='test', test_with_labels=True)
data.classes
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.model
learn.lr_find(end_lr=5)
learn.sched.plot_lr()
learn.sched.plot()
lr = 0.02
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
# helper function to get augmentations
def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]

# transformed images
ims = np.stack([get_augs() for i in range(6)])
plots(ims, rows=2)
learn.fit(lr, 2)
learn.precompute = False;
learn.fit(lr, n_cycle=3, cycle_len=1)
learn.unfreeze()
lr_diff = np.array([1e-4, 1e-3,1e-2])
learn.fit(lr_diff, n_cycle=3, cycle_len=1, cycle_mult=2)
# saving in /models
learn.save('model')
log_preds = learn.predict(is_test=True) # this gives prediction for validation set. Predictions are in log scale#
preds = np.argmax(log_preds, axis=1)    # from log probabilities to 0 or 1
probs = np.exp(log_preds[:,1])          # pr(dog)
y_true = learn.data.test_ds.y           # correct labels
print('Probabilities =\n', probs)
print('Predictions =\n', preds)
print('\nTrue Values =\n', y_true)
learn.data.test_ds.y.shape
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, preds)
plot_confusion_matrix(cm, data.classes)
def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
acts = np.array([1, 0, 0, 1])
preds = np.array([0.9, 0.1, 0.2, 0.8])
binary_loss(acts, preds)

# deleting so as to commit successfully.
!rm -rf data