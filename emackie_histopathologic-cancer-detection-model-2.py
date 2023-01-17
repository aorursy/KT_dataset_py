import numpy as np

import pandas as pd

import seaborn as sns

import os

import gc

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import random

from fastai import *

from fastai.vision import *

from torchvision.models import resnet50

from PIL import Image

from sklearn.utils import shuffle

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from skimage.restoration import estimate_sigma
data = pd.read_csv('/kaggle/input/train_labels.csv')

train_path = '/kaggle/input/train'

# quick look at the label stats

print("Value counts of data labels:")

print(data['label'].value_counts())

print("\nSample data labels:")

print(data.head())
# As per qitvision reverse the cv2 colours from bgr to rgb - we will then see the same images as directly from PCam;

# plus I wouldn't want this to affect a pretrained model's ability



def readImage(path):

    # OpenCV reads the image in bgr format by default

    bgr_img = cv2.imread(path)

    # We flip it to rgb for visualization purposes

    b,g,r = cv2.split(bgr_img)

    rgb_img = cv2.merge([r,g,b])

    return rgb_img
# random sampling

shuffled_data = shuffle(data, random_state = 27)
fig, ax = plt.subplots(2,5, figsize=(20,8))

fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20)

# Negatives

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):

    path = os.path.join(train_path, idx)

    ax[0,i].imshow(readImage(path + '.tif'))

    # Create a Rectangle patch

    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')

    ax[0,i].add_patch(box)

ax[0,0].set_ylabel('Negative samples', size='large')

# Positives

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):

    path = os.path.join(train_path, idx)

    ax[1,i].imshow(readImage(path + '.tif'))

    # Create a Rectangle patch

    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',facecolor='none', linestyle=':', capstyle='round')

    ax[1,i].add_patch(box)

ax[1,0].set_ylabel('Tumor tissue samples', size='large')
# As we count the statistics, we can check if there are any completely black or white images

dark_th = 10      # If no pixel reaches this threshold, image is considered too dark

bright_th = 245   # If no pixel is under this threshold, image is considerd too bright

too_dark_idx = []

too_bright_idx = []

bad_dtypes = []

noise_array_pos = np.array([])

noise_array_neg = np.array([])

means_pos = np.array([])

means_neg = np.array([])

stds_pos = np.array([])

stds_neg = np.array([])



N = 10000 # max length of positive and negative samples

positive_samples = []

negative_samples = []



iterable = shuffled_data["id"].head(10000)

#for i, idx in tqdm(enumerate(iterable), 'computing statistics...(220025 it total)'):

for i, idx in enumerate(iterable): 

    # What is the label

    label = shuffled_data.loc[shuffled_data["id"] == idx, "label"].values[0]

    

    # Read image

    path = os.path.join(train_path, idx)

    img = readImage(path + '.tif')

    imagearray = img.reshape(-1,3)

    

    # Check for anomylous data types:

    if img.dtype is not np.dtype('uint8'):

        bad_dtypes.append(idx)

        

    # Build separate positive and negative samples for examination later

    if label == 1 and len(positive_samples) < N:

        positive_samples.append(img)

        # Record "noise" of image

        noise_array_pos = np.append(noise_array_pos, estimate_sigma(img, multichannel=True, average_sigmas=True))

        # Record mean and std of each image

        means_pos = np.append(means_pos, np.mean(imagearray))

        stds_pos = np.append(stds_pos, np.std(imagearray))

    if label == 0 and len(negative_samples) < N:

        negative_samples.append(img)

        # Record "noise" of image

        noise_array_neg = np.append(noise_array_neg, estimate_sigma(img, multichannel=True, average_sigmas=True))

        # Record mean and std of each image

        means_neg = np.append(means_neg, np.mean(imagearray))

        stds_neg = np.append(stds_neg, np.std(imagearray))

        

    # is this too dark

    if(imagearray.max() < dark_th):

        too_dark_idx.append(idx)

        continue # do not include in statistics

    

    # is this too bright

    if(imagearray.min() > bright_th):

        too_bright_idx.append(idx)

        continue # do not include in statistics
positive_samples = np.array(positive_samples)

negative_samples = np.array(negative_samples)

gc.collect()
print("# of samples which are too dark: " + str(len(too_dark_idx)))

print("# of samples which are too light: " + str(len(too_bright_idx)))

print("# of samples which have the wrong dtype: " + str(len(bad_dtypes)))
nr_of_bins = 256 #each possible pixel value will get a bin in the following histograms

fig,axs = plt.subplots(4,2,sharey=True,figsize=(8,8),dpi=150)



#RGB channels

axs[0,0].hist(positive_samples[:,:,:,0].flatten(),bins=nr_of_bins,density=True)

axs[0,1].hist(negative_samples[:,:,:,0].flatten(),bins=nr_of_bins,density=True)

axs[1,0].hist(positive_samples[:,:,:,1].flatten(),bins=nr_of_bins,density=True)

axs[1,1].hist(negative_samples[:,:,:,1].flatten(),bins=nr_of_bins,density=True)

axs[2,0].hist(positive_samples[:,:,:,2].flatten(),bins=nr_of_bins,density=True)

axs[2,1].hist(negative_samples[:,:,:,2].flatten(),bins=nr_of_bins,density=True)



#All channels

axs[3,0].hist(positive_samples.flatten(),bins=nr_of_bins,density=True)

axs[3,1].hist(negative_samples.flatten(),bins=nr_of_bins,density=True)



#Set image labels

axs[0,0].set_title("Positive samples (N =" + str(positive_samples.shape[0]) + ")");

axs[0,1].set_title("Negative samples (N =" + str(negative_samples.shape[0]) + ")");

axs[0,1].set_ylabel("Red",rotation='horizontal',labelpad=35,fontsize=12)

axs[1,1].set_ylabel("Green",rotation='horizontal',labelpad=35,fontsize=12)

axs[2,1].set_ylabel("Blue",rotation='horizontal',labelpad=35,fontsize=12)

axs[3,1].set_ylabel("RGB",rotation='horizontal',labelpad=35,fontsize=12)

for i in range(4):

    axs[i,0].set_ylabel("Relative frequency")

axs[3,0].set_xlabel("Pixel value")

axs[3,1].set_xlabel("Pixel value")

fig.tight_layout()



nr_of_bins = 64 #we use a bit fewer bins to get a smoother image

fig,axs = plt.subplots(1,2,sharey=True, sharex = True, figsize=(8,2),dpi=150)

axs[0].hist(np.mean(positive_samples,axis=(1,2,3)),bins=nr_of_bins,density=True);

axs[1].hist(np.mean(negative_samples,axis=(1,2,3)),bins=nr_of_bins,density=True);

axs[0].set_title("Mean brightness, positive samples");

axs[1].set_title("Mean brightness, negative samples");

axs[0].set_xlabel("Image mean brightness")

axs[1].set_xlabel("Image mean brightness")

axs[0].set_ylabel("Relative frequency")

axs[1].set_ylabel("Relative frequency");
# Plot average images

fig,axs = plt.subplots(1,2,sharey=True, sharex = True, figsize=(8,2),dpi=150)

axs[0].imshow(Image.fromarray(np.mean(positive_samples,axis=(0)), 'RGB'))

axs[1].imshow(Image.fromarray(np.mean(negative_samples,axis=(0)), 'RGB'))

axs[0].set_title("Average image, positive samples");

axs[1].set_title("Average image, negative samples");
fig, axs = plt.subplots(1,2,sharey=True, sharex = True, figsize=(8,2),dpi=150)

axs[0].set_title("Correlation of pixels, positives");

axs[1].set_title("Correlation of pixels, negatives");

temp = np.mean(positive_samples[:,32:64,32:64,:], axis = 3).flatten().reshape(positive_samples.shape[0], 32*32)

temp = pd.DataFrame(data = temp, columns = [str(i) for i in range(temp.shape[1])])

sns.heatmap(temp.corr(), ax = axs[0])

temp = np.mean(negative_samples[:,32:64,32:64,:], axis = 3).flatten().reshape(negative_samples.shape[0], 32*32)

temp = pd.DataFrame(data = temp, columns = [str(i) for i in range(temp.shape[1])])

sns.heatmap(temp.corr(), ax = axs[1])

fig.show()
fig, axs = plt.subplots(1,2,sharey=True, sharex = True, figsize=(8,2),dpi=150)

axs[0].set_title("Mean vs Std, positives");

axs[1].set_title("Mean vs Std, negatives");

axs[0].scatter(means_pos, stds_pos)

axs[1].scatter(means_neg, stds_neg)
# Free memory

temp = None

axs = None

gc.collect()
nr_of_bins = 100 #we use a bit fewer bins to get a smoother image

fig,axs = plt.subplots(1,2,sharey=True, sharex = True, figsize=(8,2),dpi=150)

axs[0].hist(noise_array_pos.flatten(),bins=nr_of_bins,density=True);

axs[1].hist(noise_array_neg.flatten(),bins=nr_of_bins,density=True);

axs[0].set_title("Noise estimation of positives");

axs[1].set_title("Noise estimation of negatives");

for i in [0, 1]:

    axs[i].set_xlabel("Noise estimation")

    axs[i].set_ylabel("Relative frequency")
pca = PCA(700)

fig, axs = plt.subplots(1,2,sharey=True, sharex = True, figsize=(8,2),dpi=150)

axs[0].set_title("PCA variance, positive samples");

axs[0].set_xlabel("number of components")

axs[0].set_ylabel("cumulative explained variance")

axs[1].set_title("PCA variance, negative samples");

axs[1].set_xlabel("number of components")

axs[1].set_ylabel("cumulative explained variance")

pca.fit(positive_samples.flatten().reshape(positive_samples.shape[0], 96*96*3))

axs[0].plot(np.cumsum(pca.explained_variance_ratio_))

pca.fit(negative_samples.flatten().reshape(negative_samples.shape[0], 96*96*3))

axs[1].plot(np.cumsum(pca.explained_variance_ratio_))
# Free memory

pca = None

positive_samples, negative_samples = None, None

gc.collect()
tfms = get_transforms(do_flip=True)



bs=64 # also the default batch size

data = ImageDataBunch.from_csv(

    '/kaggle/input/', 

    ds_tfms=tfms, 

    size=224, 

    suffix=".tif",

    folder="train", 

    test="test",

    csv_labels="train_labels.csv", 

    bs=bs)



data.normalize(imagenet_stats)
def getLearner():

    return create_cnn(data, models.resnet50, pretrained=True, path='.', metrics=error_rate, ps=0.5, callback_fns=ShowGraph)



learner = getLearner()
max_lr = 2e-2

wd = 1e-4



# 1cycle policy

learner.fit_one_cycle(cyc_len=5, max_lr=max_lr, wd=wd)
preds, y, loss = learner.get_preds(with_loss=True)

# get accuracy

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc*100))

# plot learning rate of the one cycle

learner.recorder.plot_lr()
learner.save('./pcam_resnet50_frozen')
# load the baseline model

learner.load('./pcam_resnet50_frozen')



# unfreeze

learner.unfreeze()



# Fit new model with lower learning rates

learner.fit_one_cycle(cyc_len=5, max_lr=slice(4e-5,4e-4))

learner.recorder.plot_losses()
learner.save('./pcam_resnet50_finetuned')
preds,y, loss = learner.get_preds(with_loss=True)

# get accuracy

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc))
# Confusion matrix

interp = ClassificationInterpretation.from_learner(learner)

interp.plot_confusion_matrix(title='Confusion matrix')
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
losses,idxs = interp.top_losses()

interp.plot_top_losses(16, figsize=(16,16))
def pr_curve(preds, y):

    """

    Function to create precision recall curve

    

    Inputs

    ------

    preds - {tf} [prob of negative class, prob of positive class] for each observation

    y - {np.array} actual class value

    

    Outputs

    -------

    recall - {np.array} recall for different probability thresholds

    precision - {np.array} precision for different probability thresholds

    """

    temp_df = pd.DataFrame(data = np.array(preds), columns = ["negative_prob", "positive_prob"])

    temp_df["actual"] = y

    

    precision = np.array([])

    recall = np.array([])

    for threshold in range(0, 100, 10):

        threshold /= 100.

        temp_df["predicted_class"] = temp_df["positive_prob"] > threshold

        temp_df["predicted_class"] = temp_df["predicted_class"].astype(int)

        temp_df["true_positives"] = temp_df.apply(lambda x: 1 if x["actual"] == 1 and x["predicted_class"] == 1 else 0, axis = 1)

        temp_df["false_positives"] = temp_df.apply(lambda x: 1 if x["actual"] == 0 and x["predicted_class"] == 1 else 0, axis = 1)

        temp_df["true_negatives"] = temp_df.apply(lambda x: 1 if x["actual"] == 0 and x["predicted_class"] == 0 else 0, axis = 1)

        temp_df["false_negatives"] = temp_df.apply(lambda x: 1 if x["actual"] == 1 and x["predicted_class"] == 0 else 0, axis = 1)

        

        p = temp_df["true_positives"].sum() / float(temp_df["true_positives"].sum() + temp_df["false_positives"].sum())

        r = temp_df["true_positives"].sum() / float(temp_df["true_positives"].sum() + temp_df["false_negatives"].sum())

        

        precision = np.append(precision, p)

        recall = np.append(recall, r)

    

    return recall, precision
recall, precision = pr_curve(preds, y)



fig,axs = plt.subplots(1,1,sharey=True,figsize=(3,3),dpi=150)

axs.set_title("P/R Curve for Positive Class");

axs.set_xlabel("Recall")

axs.set_ylabel("Precision")



#RGB channels

axs.plot(recall, precision)