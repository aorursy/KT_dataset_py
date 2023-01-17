#put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np
from sklearn.metrics import confusion_matrix
import os, random
from shutil import copy, copytree #important for creating new working directory
import gc
#fast.ai 
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import*
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
#define train and test paths
train_path = '../input/train/'
test_path = '../input/test/'
#define numeber of train and valid images
train_samples = 10000
valid_samples = 2500
#create a list of all dog images in the train folder
train_dog = [train_path+i for i in os.listdir(train_path) if 'dog' in i]
#create a list of all cat images in the train folder
train_cat = [train_path+i for i in os.listdir(train_path) if 'cat' in i]
#shuffle images of dogs and cats
random.shuffle(train_dog)
random.shuffle(train_cat)
#only considering a number of images for both the training and valid set
train_dog_images = train_dog[:train_samples]
train_cat_images = train_cat[:train_samples]
valid_dog_images = train_dog[- valid_samples:]
valid_cat_images = train_cat[- valid_samples:]
#create new working directory 
os.makedirs('../working/dogcats/valid/cat/')
os.makedirs('../working/dogcats/valid/dog/')
os.makedirs('../working/dogcats/train/cat/')
os.makedirs('../working/dogcats/train/dog/')
#copy train images from input directory to working directory
for i in range(0,train_samples):
    shutil.copy(train_dog_images[i], '../working/dogcats/train/dog/')
    shutil.copy(train_cat_images[i], '../working/dogcats/train/cat/')
#copy valid images from input directory to working directory
for i in range(0,valid_samples):
    shutil.copy(valid_dog_images[i], '../working/dogcats/valid/dog/')
    shutil.copy(valid_cat_images[i], '../working/dogcats/valid/cat/') 
#create directory then copy test images to the new directory
shutil.copytree(test_path, '../working/dogcats/test/')
#view folders in the new directory 
os.listdir('../working/dogcats/')
len(os.listdir('../working/dogcats/test/'))
path = '../working/dogcats/'
image_size = 224
torch.cuda.is_available()
#activate an NVidia special accelerated functions for deep learning in a package called CuDNN
torch.backends.cudnn.enabled 
#create a model based on resnet34
arch = resnet34
tfms = tfms_from_model(arch, image_size, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(path, tfms=tfms, test_name='test')
learn = ConvLearner.pretrained(arch, data, precompute=True)
#find the best learning rate
lrf = learn.lr_find()
#plot learning rate against loss to determine the best learning rate
learn.sched.plot() 
#best alpha is 0.01
#fit our model
learn.fit(0.01, 2)
gc.collect()
#precompute  True >> means we are using the output of the pretrained model and passing it to the last layer
# which is a way of saving time
learn.precompute = False
#n_cycle is the number of times of resetting the learning rate back to 0.01
#cycle_len is the number of times of resetting the learning rate per an epoch
learn.fit(0.01, n_cycle=3, cycle_len=1)
#plot learning rate
learn.sched.plot_lr()
gc.collect()
#unfreeze all layers
learn.unfreeze()
#set a differential learning rate
lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()
gc.collect()
#TTA makes predictions not just on the images in your validation set, 
#but also makes predictions on a number of randomly augmented versions
valid_log_predictions,y = learn.TTA()
valid_prob_predictions = np.mean(np.exp(valid_log_predictions),0)
#calculate accuracy of valid predictions
accuracy_np(valid_prob_predictions, y)
#return a list of (1 if dog or 0 if cat)
valid_predictions = np.argmax(valid_prob_predictions, axis=1)
#plot a confussion matrix
cm = confusion_matrix(y, valid_predictions)
plot_confusion_matrix(cm, data.classes)
#predict
#log_predictions = learn.predict(is_test=True)
log_predictions,_ = learn.TTA(is_test=True)
#prob_predictions = np.exp(log_predictions[:,1])
prob_predictions = np.mean(np.exp(log_predictions),0)
prob_predictions = prob_predictions[:,1]
submission = pd.DataFrame({'id':os.listdir(f'{path}test'), 'label':prob_predictions})
! rm -rf ../working/dogcats/ #remove "working" directory
submission['id'] = submission['id'].map(lambda x: x.split('.')[0])
submission['id'] = submission['id'].astype(int)
submission = submission.sort_values('id')
submission.to_csv("submission.csv", index = False)