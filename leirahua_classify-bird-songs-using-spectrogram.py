# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
! tar xf ../input/bird-songs-pad-and-resize-spectrogram/spectrograms_resized.tar.bz2
%reload_ext autoreload

%autoreload 2

%matplotlib inline



from pathlib import Path

import matplotlib.pyplot as plt



from fastai.vision import *
data_dir = Path('../input')

label_dir = data_dir/'multilabel-bird-species-classification-nips2013/nips4b_bird_challenge_train_labels/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS'

spect_dir = Path('./spectrograms_resized')
df = pd.read_csv(label_dir/'nips4b_birdchallenge_train_labels.csv', skiprows=[0, 2])

df.tail()
df.drop(df.columns[[1, 3]],axis=1,inplace=True)  # drop colums 1 and 3

df.rename(columns={df.columns[0]:'file', df.columns[1]:'EMPTY'}, inplace=True)  # add 'EMPTY' label

df = df[:-1]  # drop last row

df.fillna(0, inplace=True)  # fill all the NaN's

df = df.astype('int32', errors='ignore') # make integer labels

df['file'] = df['file'].apply(lambda fn: str(Path(fn).with_suffix(''))) # drop .wav in the filenames



df.tail()
np.random.seed(42)

src = (ImageList.from_df(df, spect_dir, folder='train', suffix='.png')

                .split_by_rand_pct(0.2)

                .label_from_df(cols=df.columns[1:].tolist()))
tfms = get_transforms(do_flip=False, max_rotate=None, max_warp=None)

data = (src.transform(tfms, size=128)

        .databunch(num_workers=0).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9), ds_type=DatasetType.Valid)
arch = models.resnet50

acc_02 = partial(accuracy_thresh, thresh=0.2)

learn = cnn_learner(data, arch, metrics=acc_02, path='.')
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 2.29E-02

learn.fit_one_cycle(5, slice(lr))
learn.recorder.plot_losses()
learn.save('stage-1-rn50', return_path=True)
# learn.load('stage-1-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(3.02E-03, lr/5))
learn.recorder.plot_losses()
learn.save('stage-2-rn50')
data_256 = (src.transform(tfms, size=256)

            .databunch(num_workers=0).normalize(imagenet_stats))



learn.data = data_256

learn.data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 5E-03

learn.fit_one_cycle(5, slice(lr))
learn.recorder.plot_losses()
learn.save('stage-1-256-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1.58E-06, lr/5))
learn.recorder.plot_losses()
learn.save('stage-2-256-rn50')
learn.fit_one_cycle(5, slice(1.58E-06, lr/5))
learn.recorder.plot_losses()
learn.save('stage-2-256-rn50-10e')
learn.export()
test = ImageList.from_folder(spect_dir/'test/')

len(test)
predictor = load_learner('.', test=test, num_workers=0)

preds, _ = predictor.get_preds(ds_type=DatasetType.Test)

fpreds = preds[:, 1:].reshape(-1, )
import itertools

names = [f.stem for f in predictor.data.test_ds.items]

fnames = [x + '.wav_classnumber_' + str(i) for x in names for i in range(1, len(data.classes))]
test_df = pd.DataFrame({'ID':fnames, 'Probability':fpreds}, columns=['ID', 'Probability'])

test_df.to_csv('submission.csv', index=False)
! rm -r $spect_dir