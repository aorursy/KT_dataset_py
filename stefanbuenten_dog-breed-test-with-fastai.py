%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import numpy as np
import pandas as pd
import os
# make sure CUDA is available and enabled
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
comp_name = "dog_breed"
input_path = "../input/"
wd = "/kaggle/working/"
def create_symlnk(src_dir, src_name, dst_name, dst_dir=wd, target_is_dir=False):
    """
    If symbolic link does not already exist, create it by pointing dst_dir/lnk_name to src_dir/lnk_name
    """
    if not os.path.exists(dst_dir + dst_name):
        os.symlink(src=src_dir + src_name, dst = dst_dir + src_name, target_is_directory=target_is_dir)
def clean_up(wd=wd):
    """
    Delete all temporary directories and symlinks in working directory (wd)
    """
    for root, dirs, files in os.walk(wd):
        try:
            for d in dirs:
                if os.path.islink(d):
                    os.unlink(d)
                else:
                    shutil.rmtree(d)
            for f in files:
                if os.path.islink(f):
                    os.unlink(f)
                else:
                    print(f)
        except FileNotFoundError as e:
            print(e)
create_symlnk(input_path, "train", "train", target_is_dir=True)
create_symlnk(input_path, "test", "test", target_is_dir=True)
create_symlnk(input_path, "labels.csv", "labels.csv")
# perform sanity check
!ls -alh
label_df = pd.read_csv(f"{wd}labels.csv")
label_df.head()
label_df.shape
label_df.pivot_table(index="breed", aggfunc=len).sort_values("id", ascending=False)
# create validation dataset
val_idxs = get_cv_idxs(label_df.shape[0])
# define architecture
arch = resnet101
sz = 224
bs = 64
# load data
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(path=wd, folder="train", csv_fname=f"{wd}labels.csv", tfms=tfms, val_idxs=val_idxs, suffix=".jpg", test_name="test")
[print(len(e)) for e in [data.trn_ds, data.val_ds, data.test_ds]]
# look at an actual image
fn = wd + data.trn_ds.fnames[-1]
img = PIL.Image.open(fn); img
img.size
def get_data(sz=sz, bs=bs, data=data):
    """
    Load images via fastai's ImageClassifierData.from_csv() object defined as 'data' before
    Return images if size bigger than 300 pixels, else resize to 340 pixels
    """
    return data if sz > 300 else data.resize(340, new_path=wd)
data2 = get_data()
learn = ConvLearner.pretrained(arch, data2, precompute=True)
lrf = learn.lr_find()
learn.sched.plot()
# fit baseline model without data augmentation
learn.fit(1e-1, 2)
# disable precompute and fit model with data augmentation
learn.precompute=False
learn.fit(1e-1, 5, cycle_len=1)
#learn.save("224_pre")
#learn.load("224_pre")
learn.set_data(get_data(299, bs))
learn.fit(1e-1, 3, cycle_len=1)
from sklearn.metrics import log_loss

log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y), log_loss(y, probs)
#learn.save("299_pre")
#learn.load("299_pre")
log_preds_test, y_test = learn.TTA(is_test=True)
probs_test = np.mean(np.exp(log_preds_test), 0)
df = pd.DataFrame(probs_test)
df.columns = data.classes
# insert clean ids - without folder prefix and .jpg suffix - of images as first column
df.insert(0, "id", [e[5:-4] for e in data.test_ds.fnames])
df.to_csv(f"sub_{comp_name}_{str(arch.__name__)}.csv", index=False)
clean_up()