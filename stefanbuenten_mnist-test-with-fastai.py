%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
# make sure CUDA is available and enabled
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
comp_name = "digit_recognizer"
input_path = "../input/"
wd = "/kaggle/working/"
def create_symlnk(src_dir, lnk_name, dst_dir=wd, target_is_dir=False):
    """
    If symbolic link does not already exist, create it by pointing dst_dir/lnk_name to src_dir/lnk_name
    """
    if not os.path.exists(dst_dir + lnk_name):
        os.symlink(src=src_dir + lnk_name, dst = dst_dir + lnk_name, target_is_directory=target_is_dir)
create_symlnk(input_path, "train.csv")
create_symlnk(input_path, "test.csv")
# perform sanity check
!ls -alh
# load data
train_df = pd.read_csv(f"{wd}train.csv")
test_df = pd.read_csv(f"{wd}test.csv")
train_df.head()
print(train_df.shape, test_df.shape)
# create validation dataset
val_df = train_df.sample(frac=0.2, random_state=1337)
val_df.shape
# remove validation data from train dataset
train_df = train_df.drop(val_df.index)
train_df.shape
# separate labels from data
Y_train = train_df["label"]
Y_valid = val_df["label"]
X_train = train_df.drop("label", axis=1)
X_valid = val_df.drop("label", axis=1)
print(X_train.shape, X_valid.shape)
print(Y_train.shape, Y_valid.shape)
# display an actual image/digit
img = X_train.iloc[0,:].values.reshape(28,28)
plt.imshow(img, cmap="gray")
def reshape_img(matrix):
    """
    Reshape an existing 2D pandas.dataframe into 3D-numpy.ndarray
    """
    try:
        return matrix.values.reshape(-1, 28, 28)
    except AttributeError as e:
        print(e)
def add_color_channel(matrix):
    """
    Add missing color channels to previously reshaped image
    """
    matrix = np.stack((matrix, ) *3, axis = -1)
    return matrix
def convert_ndarry(matrix):
    """
    Convert pandas.series into numpy.ndarray
    """
    try:
        return matrix.values.flatten()
    except AttributeError as e:
        print(e)
# reshape data and add color channels
X_train = reshape_img(X_train)
X_train = add_color_channel(X_train)
X_valid = reshape_img(X_valid)
X_valid = add_color_channel(X_valid)
test_df = reshape_img(test_df)
test_df = add_color_channel(test_df)
# convert y_train and y_valid into proper numpy.ndarray
Y_train = convert_ndarry(Y_train)
Y_valid = convert_ndarry(Y_valid)
# run sanity checks
preprocessed_data = [X_train, Y_train, X_valid, Y_valid, test_df]
print([e.shape for e in preprocessed_data])
print([type(e) for e in preprocessed_data])
# define architecture
arch = resnet50
sz = 28
classes = np.unique(Y_train)
data = ImageClassifierData.from_arrays(path=wd, 
                                       trn=(X_train, Y_train),
                                       val=(X_valid, Y_valid),
                                       classes=Y_train,
                                       test=test_df,
                                       tfms=tfms_from_model(arch, sz))
# run learner with precompute enabled
learn = ConvLearner.pretrained(arch, data, precompute=True)
# find optimal learning rate
lrf = learn.lr_find()
# plot loss vs. learning rate
learn.sched.plot()
# fit learner
%time learn.fit(1e-2, 2)
# save model
#learn.save("28_lastlayer")
# disable precompute and unfreeze layers
learn.precompute=False
learn.unfreeze()
# define differential learning rates
lr = np.array([0.001, 0.0075, 0.01])
# retrain full model
%time learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
# save full model
#learn.save("28_all")
# get accuracy for validation set
log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y)
# predict on test set
%time log_preds_test, y_test = learn.TTA(is_test=True)
probs_test = np.mean(np.exp(log_preds_test), 0)
probs_test.shape
# create dataframe from probabilities
df = pd.DataFrame(probs_test)
# increase index by 1 to obtain proper ImageIDs
df.index += 1
# create new colum containing label with highest probability for each digit
df = df.assign(Label = df.values.argmax(axis=1))
# replicate index as dedicated ImageID column necessary for submission
df = df.assign(ImageId = df.index.values)
# drop individual probabilites
df = df.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], axis=1)
# reorder columns for submission
df = df[["ImageId", "Label"]]
# run sanity checks
df.head()
# ...
df.tail()
# ...
df.shape
# write dataframe to CSV
df.to_csv(f"sub_{comp_name}_{arch.__name__}.csv", index=False)
def clean_up():
    """
    Delete all temporary directories and symlinks in the current directory
    """
    try:
        shutil.rmtree("models")
        shutil.rmtree("tmp")
        os.unlink("test.csv")
        os.unlink("train.csv")
    except FileNotFoundError as e:
        print(e)
clean_up()
