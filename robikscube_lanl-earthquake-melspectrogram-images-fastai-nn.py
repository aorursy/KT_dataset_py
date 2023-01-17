import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import librosa.display

from librosa.feature import melspectrogram

import librosa

import random

from tqdm import tqdm_notebook

import random

from tqdm import tqdm

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
# Train labels can be found in this CSV

train_labels = pd.read_csv('../input/lanl-earthquake-spectrogram-images/training_labels.csv')

train_labels.head()
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))

img1 = mpimg.imread('../input/lanl-earthquake-spectrogram-images/train_images_no_overlap/train_images_v3/train_0.png')

img2 = mpimg.imread('../input/lanl-earthquake-spectrogram-images/train_images_no_overlap/train_images_v3/train_100.png')

img3 = mpimg.imread('../input/lanl-earthquake-spectrogram-images/train_images_no_overlap/train_images_v3/train_200.png')

img4 = mpimg.imread('../input/lanl-earthquake-spectrogram-images/train_images_no_overlap/train_images_v3/train_300.png')

img5 = mpimg.imread('../input/lanl-earthquake-spectrogram-images/train_images_no_overlap/train_images_v3/train_400.png')

ax1.imshow(img1)

ax1.set_title('TTF - {:0.4f}'.format(train_labels.loc[train_labels['seg_id'] == 'train_0']['target'].values[0]), fontsize=25)

ax2.imshow(img2)

ax2.set_title('TTF - {:0.4f}'.format(train_labels.loc[train_labels['seg_id'] == 'train_100']['target'].values[0]), fontsize=25)

ax3.imshow(img3)

ax3.set_title('TTF - {:0.4f}'.format(train_labels.loc[train_labels['seg_id'] == 'train_200']['target'].values[0]), fontsize=25)

ax4.imshow(img4)

ax4.set_title('TTF - {:0.4f}'.format(train_labels.loc[train_labels['seg_id'] == 'train_300']['target'].values[0]), fontsize=25)

ax5.imshow(img5)

ax5.set_title('TTF - {:0.4f}'.format(train_labels.loc[train_labels['seg_id'] == 'train_400']['target'].values[0]), fontsize=25)

ax1.axis('off')

ax2.axis('off')

ax3.axis('off')

ax4.axis('off')

ax5.axis('off')

plt.tight_layout()

plt.show()
seg_00030f = pd.read_csv('../input/LANL-Earthquake-Prediction/test/seg_00030f.csv')

y = seg_00030f['acoustic_data'].astype('float').values
def plot_spectrogram(y, imgdir=None, imgname=None, plot=True, sr=10000, n_mels=1000, log_tf=True, vmin=-100, vmax=0):

    # Let's make and display a mel-scaled power (energy-squared) spectrogram

    #y = np.array([float(x) for x in df['acoustic_data'].values])

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)



    # Convert to log scale (dB). We'll use the peak power (max) as reference.

    if log_tf:

        S = librosa.power_to_db(S, ref=np.max)

    

    if plot:

        # Make a new figure

        plt.figure(figsize=(15,5))

        plt.imshow(S)

        # draw a color bar

        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact

        plt.tight_layout()

        plt.axis('off')

    if imgname is not None:

        plt.imsave('{}/{}.png'.format(imgdir, imgname), S)

        plt.clf()

        plt.close()

    return
# Plot an example using this function.

plot_spectrogram(y)
%reload_ext autoreload

%autoreload 2

%matplotlib inline



# I personally hate that fastai encourages using `import *`

# That is how it is taught in the course

from fastai import *

from fastai.vision import *
bs = 10

train_labels['path'] = '../input/lanl-earthquake-spectrogram-images/train_images_no_overlap/train_images_v3/' + train_labels['seg_id'] + '.png'

#valid_idx = train_labels[:10000].loc[train_labels['quake_number'].isin([1, 5, 8])].index

#train_idx = train_labels[:10000].loc[~train_labels['quake_number'].isin([1, 5, 8])].index

data = (ImageList.from_df(train_labels[:-1], path='./', cols='path')

        #.split_by_idxs(train_idx=train_idx, valid_idx=valid_idx)

        .split_by_rand_pct(0.1)

        .label_from_df('target', label_cls=FloatList)

        #.transform(get_transforms(), size=255)

        .databunch(bs=bs))
data.show_batch(rows=3, figsize=(7,6))
def mean_absolute_error(pred:Tensor, targ:Tensor)->Rank0Tensor:

    "Mean absolute error between `pred` and `targ`."

    pred,targ = flatten_check(pred,targ)

    return torch.abs(targ - pred).mean()



learn = cnn_learner(data, models.resnet50, metrics=mean_absolute_error)

learn.fit_one_cycle(4, 0.01)
# Plot train vs valid loss

fig = learn.recorder.plot_losses(return_fig=True)

fig.set_size_inches(15,5)
# Unfreeze the model and search for a good learning rate

learn.unfreeze()

learn.lr_find()

fig = learn.recorder.plot(return_fig=True)

fig.set_size_inches(15,5)
learn.fit_one_cycle(2, slice(1e-6, 3e-3/10))

learn.save('cnn-step1')
# Export the model

learn.export()
# We can see there is now an export.pkl file that we've saved

!ls -l
ss = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

test = ImageList.from_df(ss, '../input/lanl-earthquake-spectrogram-images/test_images/test_images_v3/', cols='seg_id', suffix='.png')

learn = load_learner('./', test=test)

learn.load('cnn-step1')

preds = learn.get_preds(ds_type=DatasetType.Test)
# Save the time to failure

ss['time_to_failure'] = [float(x) for x in preds[0]]
ss.head()
# Cap the minimum and maximum time to failure values

ss.loc[ss['time_to_failure'] < 0, 'time_to_failure'] = 0

ss.loc[ss['time_to_failure'] > 12, 'time_to_failure'] = 12
ss.plot(kind='hist', bins=100, figsize=(15, 5), title='Distribution of predictions on the Test Set')

plt.show()
# Save our predictions

ss.to_csv('submission.csv', index=False)