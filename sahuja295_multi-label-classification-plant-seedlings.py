import os ; import shutil ; import sys ; import warnings
import numpy as np ; import pandas as pd
import matplotlib.pyplot as plt ; import seaborn as sns
%matplotlib inline
%reload_ext autoreload
%autoreload 2
from PIL import Image
from scipy.misc import imread
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from fastai.conv_learner import *
from fastai.imports import *
!ls "../"
!rm -r "../data"
!mkdir "../data"
!mkdir "../data/train"
!mkdir "../data/test"
!ls "../"
INPUT_PATH = "../input/"
PATH="../data/"
TRAIN_PATH="train"
TEST_PATH="test"

# Helper Function to move files from sub-folders to /train folder
classes = os.listdir(INPUT_PATH + TRAIN_PATH)
print(classes)
labels_array = []
for subdir in os.listdir(INPUT_PATH + TRAIN_PATH):    
    files = os.listdir(INPUT_PATH + TRAIN_PATH + '/' + subdir)    
    category = [subdir] * len(files)     
    array = list(zip(files,category))    
    labels_array.extend(array)    
    for file in files:
        shutil.copy(INPUT_PATH + TRAIN_PATH + '/' + subdir + '/' + file, PATH + TRAIN_PATH + '/' + file)
        #shutil.rmtree(PATH + TRAIN_PATH + '/' + subdir)
labels_array = np.array(labels_array)
print(labels_array.shape)
print(labels_array[:5])
df = pd.DataFrame(labels_array)
df.columns=["file","category"]
df.replace(to_replace=' ', value='_',inplace=True,regex=True)
df.to_csv(f'{PATH}/train_labels.csv',index=False,header=True)

files = os.listdir(INPUT_PATH + TEST_PATH)
for file in files:
    shutil.copy(INPUT_PATH + TEST_PATH + '/' + file, PATH + TEST_PATH + '/' + file)
labels_file=f'{PATH}train_labels.csv' #no_labels_file
train_labels=pd.read_csv(labels_file,low_memory=False)
print(train_labels.head(10))
print(os.listdir(PATH + TRAIN_PATH))
TMP_PATH = 'tmp'
MODEL_PATH = 'model'
f_model = resnet34 ; f_model_str = 'resnet34'
n = len(list(open(labels_file)))-1
val_idxs = get_cv_idxs(n,val_pct=0.2)
def get_data(sz,bs=64):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, max_zoom=1.05)
    return ImageClassifierData.from_csv(
                                        PATH,
                                        TRAIN_PATH,
                                        labels_file,
                                        tfms=tfms,
                                        bs=bs,
                                        val_idxs=val_idxs,
                                        test_name=TEST_PATH)
data = get_data(64)
img = PIL.Image.open(PATH + data.trn_ds.fnames[0])
plt.imshow(img)
plt.show()
#x,y = next(iter(data.val_dl))
#plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.0)
# to brighten the image
# plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4) 
sz=64
data = get_data(sz)
learn = ConvLearner.pretrained(f_model, data,
                               tmp_name=TMP_PATH, models_name=MODEL_PATH,precompute=True)
lrf=learn.lr_find()
learn.sched.plot()
lr = 0.1
lrs = np.array([lr/9,lr/3,lr])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{f_model_str}_{sz}_freeze')
print("train={},val={},test={}".format(len(data.trn_ds),len(data.val_ds),len(data.test_ds)))
log_preds = learn.predict()
preds = np.argmax(log_preds,axis=1) ; print(preds[:5])
probs = np.exp(log_preds[:,1]) ; print(probs[:5])

log_preds_test = learn.predict(is_test=True)
preds_test = np.argmax(log_preds_test,axis=1) ; print(preds_test[:5])
probs_test=np.exp(log_preds_test[:,1]) ; print(probs_test[:5])
def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], min(len(preds), 4), replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8)) if len(imgs)>0 else print('Not Found.')

def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)

plot_val_with_title(rand_by_correct(True), "Correctly classified")
plot_val_with_title(rand_by_correct(False), "Incorrectly classified")
plot_val_with_title(most_by_correct(0, True), "Most correct doesnt contain invasive hydrangea")
plot_val_with_title(most_by_correct(1, True), "Most correct contain invasive hydrangea")
plot_val_with_title(most_by_correct(0, False), "Most incorrect doesnt contain invasive hydrangea")
plot_val_with_title(most_by_correct(1, False), "Most incorrect contain invasive hydrangea")
most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")
learn.precompute=False
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
learn.save(f'{f_model_str}_{sz}_unfreeze')
sz=128
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
learn.save(f'{f_model_str}_{sz}_freeze')
sz=128
learn.precompute=False
learn.set_data(get_data(sz))
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
learn.save(f'{f_model_str}_{sz}_unfreeze')
sz=256
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
learn.save(f'{f_model_str}_{sz}_freeze')
sz=256
learn.set_data(get_data(sz))
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
learn.save(f'{f_model_str}_{sz}_unfreeze')
log_preds_tta,y1 = learn.TTA() ; print(log_preds_tta.shape) ; print (log_preds_tta[:5])
probs_tta = np.mean(np.exp(log_preds_tta),0) ; print(probs_tta.shape) ; print (probs_tta[:5])
accuracy_np(probs_tta, y1)
log_preds_test = learn.predict(is_test=True) ; print(log_preds_test.shape) ; print (log_preds_test[:5])
preds_test = np.argmax(log_preds_test,axis=1) ; print(preds_test.shape) ; print(preds_test[:5])
probs_test = np.exp(log_preds_test[:,1]) ; print(probs_test.shape) ; print(probs_test[:5])
log_preds_test_tta, _ = learn.TTA(is_test=True) ; print(log_preds_test_tta.shape) ; print(log_preds_test_tta[:5])
mean_logpreds_test_tta = np.mean(log_preds_test_tta, 0) ; print(mean_logpreds_test_tta.shape) ; print(mean_logpreds_test_tta[:5])
max_preds_test_tta = np.argmax(mean_logpreds_test_tta,axis=1) ; print(max_preds_test_tta.shape) ; print(max_preds_test_tta[:5])
probs_test_tta=np.exp(mean_logpreds_test_tta[:,1]) ; print(probs_test_tta.shape) ; print(probs_test_tta[:5])
class_preds = [data.classes[index_pred] for index_pred in max_preds_test_tta]
fnames_nopath = [(fname[len(TEST_PATH)+1:]) for fname in data.test_ds.fnames]
fname_preds = list(zip(fnames_nopath, class_preds))
df = pd.DataFrame(fname_preds)
df.columns = ["file", "species"]
df.replace(to_replace='_', value=' ',inplace=True,regex=True)
#df.sort_values(['file'], ascending=[True],inplace=True)
df.to_csv(f'subm.csv', index=False)
df_read = pd.read_csv(f'subm.csv')
df_read.head(5)