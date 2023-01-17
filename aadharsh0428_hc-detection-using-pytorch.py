import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import random

from sklearn.utils import shuffle

from tqdm import tqdm_notebook

data = pd.read_csv('/kaggle/input/train_labels.csv')

train_path = '/kaggle/input/train/'

test_path = '/kaggle/input/test/'

data['label'].value_counts()
def readImage(path):

    bgr_img = cv2.imread(path)

    b,g,r = cv2.split(bgr_img)

    rgb_img = cv2.merge([r,g,b])

    return rgb_img


shuffled_data = shuffle(data)

fig, ax = plt.subplots(2,5, figsize=(20,8))

fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20)

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):

    path = os.path.join(train_path, idx)

    ax[0,i].imshow(readImage(path + '.tif'))

    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')

    ax[0,i].add_patch(box)

ax[0,0].set_ylabel('Negative samples', size='large')

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):

    path = os.path.join(train_path, idx)

    ax[1,i].imshow(readImage(path + '.tif'))

    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',facecolor='none', linestyle=':', capstyle='round')

    ax[1,i].add_patch(box)

ax[1,0].set_ylabel('Tumor tissue samples', size='large')
import random

import imgaug as ia

from imgaug import augmenters as iaa

import numpy as np

ORIGINAL_SIZE = 96 

CROP_SIZE = 90          

RANDOM_ROTATION = 3    

RANDOM_SHIFT = 2         

RANDOM_BRIGHTNESS = 7  

RANDOM_CONTRAST = 5   

RANDOM_90_DEG_TURN = 1

def readCroppedImage(path, augmentations = True):

    bgr_img = cv2.imread(path)

    b,g,r = cv2.split(bgr_img)

    rgb_img = cv2.merge([r,g,b])

    if(not augmentations):

        return rgb_img / 255

    rotation = random.randint(-RANDOM_ROTATION,RANDOM_ROTATION)

    if(RANDOM_90_DEG_TURN == 1):

        rotation += random.randint(-1,1) * 90

    M = cv2.getRotationMatrix2D((48,48),rotation,1) 

    rgb_img = cv2.warpAffine(rgb_img,M,(96,96))

    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)

    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)

    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2

    end_crop = start_crop + CROP_SIZE

    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255

    flip_hor = bool(random.getrandbits(1))

    flip_ver = bool(random.getrandbits(1))

    if(flip_hor):

        rgb_img = rgb_img[:, ::-1]

    if(flip_ver):

        rgb_img = rgb_img[::-1, :]

    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.

    rgb_img = rgb_img + br

    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.

    rgb_img = rgb_img * cr

    rgb_img = np.clip(rgb_img, 0, 1.0)

    return rgb_img
fig, ax = plt.subplots(2,5, figsize=(20,8))

fig.suptitle('Cropped histopathologic scans of lymph node sections',fontsize=20)

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):

    path = os.path.join(train_path, idx)

    ax[0,i].imshow(readCroppedImage(path + '.tif'))

ax[0,0].set_ylabel('Negative samples', size='large')

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):

    path = os.path.join(train_path, idx)

    ax[1,i].imshow(readCroppedImage(path + '.tif'))

ax[1,0].set_ylabel('Tumor tissue samples', size='large')


dark_th = 10 / 255      

bright_th = 245 / 255   

too_dark_idx = []

too_bright_idx = []

x_tot = np.zeros(3)

x2_tot = np.zeros(3)

counted_ones = 0

for i, idx in tqdm_notebook(enumerate(shuffled_data['id']), 'computing statistics...(220025 it total)'):

    path = os.path.join(train_path, idx)

    imagearray = readCroppedImage(path + '.tif', augmentations = False).reshape(-1,3)

    if(imagearray.max() < dark_th):

        too_dark_idx.append(idx)

        continue 

    if(imagearray.min() > bright_th):

        too_bright_idx.append(idx)

        continue 

    x_tot += imagearray.mean(axis=0)

    x2_tot += (imagearray**2).mean(axis=0)

    counted_ones += 1

channel_avr = x_tot/counted_ones

channel_std = np.sqrt(x2_tot/counted_ones - channel_avr**2)

channel_avr,channel_std

print('There was {0} extremely dark image'.format(len(too_dark_idx)))

print('and {0} extremely bright images'.format(len(too_bright_idx)))

print('Dark one:')

print(too_dark_idx)

print('Bright ones:')

print(too_bright_idx)
fig, ax = plt.subplots(2,6, figsize=(25,9))

fig.suptitle('Almost completely black or white images',fontsize=20)

i = 0

for idx in np.asarray(too_dark_idx)[:min(6, len(too_dark_idx))]:

    lbl = shuffled_data[shuffled_data['id'] == idx]['label'].values[0]

    path = os.path.join(train_path, idx)

    ax[0,i].imshow(readCroppedImage(path + '.tif', augmentations = False))

    ax[0,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8)

    i += 1

ax[0,0].set_ylabel('Extremely dark images', size='large')

for j in range(min(6, len(too_dark_idx)), 6):

    ax[0,j].axis('off') 

i = 0

for idx in np.asarray(too_bright_idx)[:min(6, len(too_bright_idx))]:

    lbl = shuffled_data[shuffled_data['id'] == idx]['label'].values[0]

    path = os.path.join(train_path, idx)

    ax[1,i].imshow(readCroppedImage(path + '.tif', augmentations = False))

    ax[1,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8)

    i += 1

ax[1,0].set_ylabel('Extremely bright images', size='large')

for j in range(min(6, len(too_bright_idx)), 6):

    ax[1,j].axis('off')
from sklearn.model_selection import train_test_split

train_df = data.set_index('id')

train_names = train_df.index.values

train_labels = np.asarray(train_df['label'].values)

tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.2, stratify=train_labels, random_state=123)
from fastai import *

from fastai.vision import *

from torchvision.models import *  

arch = squeezenet1_1                 

BATCH_SIZE = 128                    

sz = CROP_SIZE                     

MODEL_PATH = str(arch).split()[1]   


train_dict = {'name': train_path + train_names, 'label': train_labels}

df = pd.DataFrame(data=train_dict)

test_names = []

for f in os.listdir(test_path):

    test_names.append(test_path + f)

df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])


class MyImageItemList(ImageList):

    def open(self, fn:PathOrStr)->Image:

        img = readCroppedImage(fn.replace('/./','').replace('//','/'))

        return vision.Image(px=pil2tensor(img, np.float32))


imgDataBunch = (MyImageItemList.from_df(path='/', df=df, suffix='.tif')

        .split_by_idx(val_idx)

        .label_from_df(cols='label')

        .add_test(MyImageItemList.from_df(path='/', df=df_test))

        .transform(tfms=[[],[]], size=sz)

        .databunch(bs=BATCH_SIZE)

        .normalize([tensor([0.702447, 0.546243, 0.696453]), tensor([0.238893, 0.282094, 0.216251])])

       )
imgDataBunch.show_batch(rows=2, figsize=(4,4))


def getLearner():

    return create_cnn(imgDataBunch, arch, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

learner = getLearner()
lrs = []

losses = []

wds = []

iter_count = 600



# WEIGHT DECAY = 1e-6

learner.lr_find(wd=1e-6, num_it=iter_count)

lrs.append(learner.recorder.lrs)

losses.append(learner.recorder.losses)

wds.append('1e-6')

learner = getLearner() #reset learner - this gets more consistent starting conditions



# WEIGHT DECAY = 1e-4

learner.lr_find(wd=1e-4, num_it=iter_count)

lrs.append(learner.recorder.lrs)

losses.append(learner.recorder.losses)

wds.append('1e-4')

learner = getLearner() #reset learner - this gets more consistent starting conditions



# WEIGHT DECAY = 1e-2

learner.lr_find(wd=1e-2, num_it=iter_count)

lrs.append(learner.recorder.lrs)

losses.append(learner.recorder.losses)

wds.append('1e-2')

learner = getLearner() #reset learner
# Plot weight decays

_, ax = plt.subplots(1,1)

min_y = 0.5

max_y = 0.55

for i in range(len(losses)):

    ax.plot(lrs[i], losses[i])

    min_y = min(np.asarray(losses[i]).min(), min_y)

ax.set_ylabel("Loss")

ax.set_xlabel("Learning Rate")

ax.set_xscale('log')

#ax ranges may need some tuning with different model architectures 

ax.set_xlim((1e-3,3e-1))

ax.set_ylim((min_y - 0.02,max_y))

ax.legend(wds)

ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
max_lr = 2e-2

wd = 1e-4

# 1cycle policy

learner.fit_one_cycle(cyc_len=8, max_lr=max_lr, wd=wd)
learner.recorder.plot_lr()
learner.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learner)

interp.plot_confusion_matrix(title='Confusion matrix')
learner.save(MODEL_PATH + '_stage1')
# load the baseline model

learner.load(MODEL_PATH + '_stage1')



# unfreeze and run learning rate finder again

learner.unfreeze()

learner.lr_find(wd=wd)



# plot learning rate finder results

learner.recorder.plot()
learner.fit_one_cycle(cyc_len=12, max_lr=slice(4e-5,4e-4))
learner.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learner)

interp.plot_confusion_matrix(title='Confusion matrix')
learner.save(MODEL_PATH + '_stage2')
preds,y, loss = learner.get_preds(with_loss=True)

# get accuracy

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc))
from random import randint



def plot_overview(interp:ClassificationInterpretation, classes=['Negative','Tumor']):

    # top losses will return all validation losses and indexes sorted by the largest first

    tl_val,tl_idx = interp.top_losses()

    #classes = interp.data.classes

    fig, ax = plt.subplots(3,4, figsize=(16,12))

    fig.suptitle('Predicted / Actual / Loss / Probability',fontsize=20)

    # Random

    for i in range(4):

        random_index = randint(0,len(tl_idx))

        idx = tl_idx[random_index]

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        im = image2np(im.data)

        cl = int(cl)

        ax[0,i].imshow(im)

        ax[0,i].set_xticks([])

        ax[0,i].set_yticks([])

        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)

    # Most incorrect or top losses

    for i in range(4):

        idx = tl_idx[i]

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        im = image2np(im.data)

        ax[1,i].imshow(im)

        ax[1,i].set_xticks([])

        ax[1,i].set_yticks([])

        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)

    # Most correct or least losses

    for i in range(4):

        idx = tl_idx[len(tl_idx) - i - 1]

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        im = image2np(im.data)

        ax[2,i].imshow(im)

        ax[2,i].set_xticks([])

        ax[2,i].set_yticks([])

        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)
plot_overview(interp, ['Negative','Tumor'])
from fastai.callbacks.hooks import *



# hook into forward pass

def hooked_backward(m, oneBatch, cat):

    # we hook into the convolutional part = m[0] of the model

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(oneBatch)

            preds[0,int(cat)].backward()

    return hook_a,hook_g
# We can create a utility function for getting a validation image with an activation map

def getHeatmap(val_index):

    """Returns the validation set image and the activation map"""

    # this gets the model

    m = learner.model.eval()

    tensorImg,cl = imgDataBunch.valid_ds[val_index]

    # create a batch from the one image

    oneBatch,_ = imgDataBunch.one_item(tensorImg)

    oneBatch_im = vision.Image(imgDataBunch.denorm(oneBatch)[0])

    # convert batch tensor image to grayscale image with opencv

    cvIm = cv2.cvtColor(image2np(oneBatch_im.data), cv2.COLOR_RGB2GRAY)

    # attach hooks

    hook_a,hook_g = hooked_backward(m, oneBatch, cl)

    # get convolutional activations and average from channels

    acts = hook_a.stored[0].cpu()

    #avg_acts = acts.mean(0)



    # Grad-CAM

    grad = hook_g.stored[0][0].cpu()

    grad_chan = grad.mean(1).mean(1)

    grad.shape,grad_chan.shape

    mult = (acts*grad_chan[...,None,None]).mean(0)

    return mult, cvIm
# Then, modify our plotting func a bit

def plot_heatmap_overview(interp:ClassificationInterpretation, classes=['Negative','Tumor']):

    # top losses will return all validation losses and indexes sorted by the largest first

    tl_val,tl_idx = interp.top_losses()

    #classes = interp.data.classes

    fig, ax = plt.subplots(3,4, figsize=(16,12))

    fig.suptitle('Grad-CAM\nPredicted / Actual / Loss / Probability',fontsize=20)

    # Random

    for i in range(4):

        random_index = randint(0,len(tl_idx))

        idx = tl_idx[random_index]

        act, im = getHeatmap(idx)

        H,W = im.shape

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        ax[0,i].imshow(im)

        ax[0,i].imshow(im, cmap=plt.cm.gray)

        ax[0,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        ax[0,i].set_xticks([])

        ax[0,i].set_yticks([])

        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)

    # Most incorrect or top losses

    for i in range(4):

        idx = tl_idx[i]

        act, im = getHeatmap(idx)

        H,W = im.shape

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        ax[1,i].imshow(im)

        ax[1,i].imshow(im, cmap=plt.cm.gray)

        ax[1,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        ax[1,i].set_xticks([])

        ax[1,i].set_yticks([])

        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)

    # Most correct or least losses

    for i in range(4):

        idx = tl_idx[len(tl_idx) - i - 1]

        act, im = getHeatmap(idx)

        H,W = im.shape

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        ax[2,i].imshow(im)

        ax[2,i].imshow(im, cmap=plt.cm.gray)

        ax[2,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        ax[2,i].set_xticks([])

        ax[2,i].set_yticks([])

        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)
plot_heatmap_overview(interp, ['Negative','Tumor'])

from sklearn.metrics import roc_curve, auc

# probs from log preds

probs = np.exp(preds[:,1])

# Compute ROC curve

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)



# Compute ROC area

roc_auc = auc(fpr, tpr)

print('ROC area is {0}'.format(roc_auc))

plt.figure()

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
# make sure we have the best performing model stage loaded

learner.load(MODEL_PATH + '_stage2')



# Fastai has a function for this but we don't want the additional augmentations it does (our image loader has augmentations) so we just use the get_preds

#preds_test,y_test=learner.TTA(ds_type=DatasetType.Test)



# We do a fair number of iterations to cover different combinations of flips and rotations.

# The predictions are then averaged.

n_aug = 12

preds_n_avg = np.zeros((len(learner.data.test_ds.items),2))

for n in tqdm_notebook(range(n_aug), 'Running TTA...'):

    preds,y = learner.get_preds(ds_type=DatasetType.Test, with_loss=False)

    preds_n_avg = np.sum([preds_n_avg, preds.numpy()], axis=0)

preds_n_avg = preds_n_avg / n_aug
print('Negative and Tumor Probabilities: ' + str(preds_n_avg[0]))

tumor_preds = preds_n_avg[:, 1]

print('Tumor probability: ' + str(tumor_preds[0]))

# If we wanted to get the predicted class, argmax would get the index of the max

class_preds = np.argmax(preds_n_avg, axis=1)

classes = ['Negative','Tumor']

print('Class prediction: ' + classes[class_preds[0]])
# get test id's from the sample_submission.csv and keep their original order

SAMPLE_SUB = '/kaggle/input/sample_submission.csv'

sample_df = pd.read_csv(SAMPLE_SUB)

sample_list = list(sample_df.id)



# List of tumor preds. 

# These are in the order of our test dataset and not necessarily in the same order as in sample_submission

pred_list = [p for p in tumor_preds]



# To know the id's, we create a dict of id:pred

pred_dic = dict((key, value) for (key, value) in zip(learner.data.test_ds.items, pred_list))



# Now, we can create a new list with the same order as in sample_submission

pred_list_cor = [pred_dic['///kaggle/input/test/' + id + '.tif'] for id in sample_list]



# Next, a Pandas dataframe with id and label columns.

df_sub = pd.DataFrame({'id':sample_list,'label':pred_list_cor})



# Export to csv

df_sub.to_csv('{0}_submission.csv'.format(MODEL_PATH), header=True, index=False)
df_sub.head(10)