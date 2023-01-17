!rm -rf utils.py

!wget https://raw.githubusercontent.com/sevenfx/fastai_audio/master/notebooks/utils.py
%matplotlib inline

import os

from pathlib import Path

from IPython.display import Audio

import librosa

import librosa.display

import matplotlib.pyplot as plt

import numpy as np

from utils import read_file, transform_path
!rm -rf free-spoken-digit-dataset-master master.zip

!wget https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip

!unzip -q master.zip

!rm -rf master.zip

!ls
AUDIO_DIR = Path('free-spoken-digit-dataset-master/recordings')

IMG_DIR = Path('imgs')

!mkdir {IMG_DIR} -p
fnames = os.listdir(str(AUDIO_DIR))

len(fnames), fnames[:5]
fn = fnames[94]

print(fn)

Audio(str(AUDIO_DIR/fn))
# ??read_file
x, sr = read_file(fn, AUDIO_DIR)

x.shape, sr, x.dtype
def log_mel_spec_tfm(fname, src_path, dst_path):

    x, sample_rate = read_file(fname, src_path)

    

    n_fft = 1024

    hop_length = 256

    n_mels = 40

    fmin = 20

    fmax = sample_rate / 2 

    

    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft, 

                                                    hop_length=hop_length, 

                                                    n_mels=n_mels, power=2.0, 

                                                    fmin=fmin, fmax=fmax)

    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)

    dst_fname = dst_path / (fname[:-4] + '.png')

    plt.imsave(dst_fname, mel_spec_db)
log_mel_spec_tfm(fn, AUDIO_DIR, IMG_DIR)

img = plt.imread(str(IMG_DIR/(fn[:-4] + '.png')))

plt.imshow(img, origin='lower');
transform_path(AUDIO_DIR, IMG_DIR, log_mel_spec_tfm, fnames=fnames, delete=True)
os.listdir(str(IMG_DIR))[:10]
import fastai

fastai.__version__
from fastai.vision import *
digit_pattern = r'(\d+)_\w+_\d+.png$'
data = (ImageList.from_folder(IMG_DIR)

        .split_by_rand_pct(0.2)

        #.split_by_valid_func(lambda fname: 'nicolas' in str(fname))

        .label_from_re(digit_pattern)

        .transform(size=(128,64))

        .databunch())

data.c, data.classes
# Shape of batch

xs, ys = data.one_batch()

xs.shape, ys.shape
# Stats

xs.min(), xs.max(), xs.mean(), xs.std()
# Sample batch

data.show_batch(4, figsize=(5,9), hide_axis=False)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(4)
learn.unfreeze()

learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10, 10), dpi=60)
# Clean up (Kaggle)

# !rm -rf {AUDIO_DIR}

# !rm -rf {IMG_DIR}
import jovian
jovian.commit()