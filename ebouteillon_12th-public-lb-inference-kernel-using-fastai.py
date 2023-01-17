import os

from pathlib import Path

import pickle

import random

import time

from io import StringIO

from csv import writer

import gc



import numpy as np

import pandas as pd

import librosa

import librosa.display

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import IPython

import IPython.display

# import PIL



import torch

import torch.nn as nn

import torch.nn.functional as F



from fastai import *

from fastai.vision import *

from fastai.vision.data import *
models_list = (

    # trained weights of CNN-model-1 inspired by mhiro2

    (Path('../input/freesoundaudiotagging2019ebouteillonsolution/fat2019ssl4multistage/work/'), 'stage-2_fold-{fold}.pkl'),

    (Path('../input/freesoundaudiotagging2019ebouteillonsolution/fat2019ssl4multistage/work/'), 'stage-10_fold-{fold}.pkl'),

    (Path('../input/freesoundaudiotagging2019ebouteillonsolution/fat2019ssl4multistage/work/'), 'stage-11_fold-{fold}.pkl'),

    # trained weights of VGG-16

    (Path('../input/freesoundaudiotagging2019ebouteillonsolution/fat2019ssl8vgg16full/work'), 'stage-2_fold-{fold}.pkl'),

    (Path('../input/freesoundaudiotagging2019ebouteillonsolution/fat2019ssl8vgg16full/work'), 'stage-10_fold-{fold}.pkl'),

    (Path('../input/freesoundaudiotagging2019ebouteillonsolution/fat2019ssl8vgg16full/work'), 'stage-11_fold-{fold}.pkl'),

)
PREDICTION_WINDOW_SHIFT = 48  # predict every PREDICTION_WINDOW_SHIFT time sample

n_splits = 10

DATA = Path('../input/freesound-audio-tagging-2019')

DATA_TEST = DATA/'test'

CSV_SUBMISSION = DATA/'sample_submission.csv'

test_df = pd.read_csv(CSV_SUBMISSION)
def read_audio(conf, pathname, trim_long_data):

    y, sr = librosa.load(pathname, sr=conf.sampling_rate)

    # trim silence

    if 0 < len(y): # workaround: 0 length causes error

        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    # make it unified length to conf.samples

    if len(y) > conf.samples: # long enough

        if trim_long_data:

            y = y[0:0+conf.samples]

    else: # pad blank

        padding = conf.samples - len(y)    # add padding at both ends

        offset = padding // 2

        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

    return y



def audio_to_melspectrogram(conf, audio):

    spectrogram = librosa.feature.melspectrogram(audio, 

                                                 sr=conf.sampling_rate,

                                                 n_mels=conf.n_mels,

                                                 hop_length=conf.hop_length,

                                                 n_fft=conf.n_fft,

                                                 fmin=conf.fmin,

                                                 fmax=conf.fmax)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    return spectrogram



def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):

    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 

                             sr=conf.sampling_rate, hop_length=conf.hop_length,

                            fmin=conf.fmin, fmax=conf.fmax)

    plt.colorbar(format='%+2.0f dB')

    plt.title(title)

    plt.show()



def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):

    x = read_audio(conf, pathname, trim_long_data)

    mels = audio_to_melspectrogram(conf, x)

    if debug_display:

        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))

        show_melspectrogram(conf, mels)

    return mels





class conf:

    # Preprocessing settings

    sampling_rate = 44100

    duration = 2

    hop_length = 347*duration # to make time steps 128

    fmin = 20

    fmax = sampling_rate // 2

    n_mels = 128

    n_fft = n_mels * 20

    samples = sampling_rate * duration





def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    # Stack X as [X,X,X]

    X = np.stack([X, X, X], axis=-1)



    # Standardize

    mean = mean or X.mean()

    std = std or X.std()

    Xstd = (X - mean) / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Scale to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V



def convert_wav_to_image(df, source, img_dest):

    print(f'Converting {source} -> {img_dest}')

    X = []

    for i, row in tqdm_notebook(df.iterrows(), total=df.shape[0]):

        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)

        x_color = mono_to_color(x)

        X.append(x_color)

    return X
X_test = convert_wav_to_image(test_df, source=DATA_TEST, img_dest=None)
# This implemented my new data augmentation technique 'SpecMix', see my github for more details :)

class MyMixUpCallback(LearnerCallback):

    def __init__(self, learn:Learner):

        super().__init__(learn)

        

class Lwlrap(Callback):

    def on_epoch_begin(self, **kwargs):

        pass
# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU

def _one_sample_positive_class_precisions(scores, truth):

    """Calculate precisions for each true class for a single sample.



    Args:

      scores: np.array of (num_classes,) giving the individual classifier scores.

      truth: np.array of (num_classes,) bools indicating which classes are true.



    Returns:

      pos_class_indices: np.array of indices of the true classes for this sample.

      pos_class_precisions: np.array of precisions corresponding to each of those

        classes.

    """

    num_classes = scores.shape[0]

    pos_class_indices = np.flatnonzero(truth > 0)

    # Only calculate precisions if there are some true classes.

    if not len(pos_class_indices):

        return pos_class_indices, np.zeros(0)

    # Retrieval list of classes for this sample.

    retrieved_classes = np.argsort(scores)[::-1]

    # class_rankings[top_scoring_class_index] == 0 etc.

    class_rankings = np.zeros(num_classes, dtype=np.int)

    class_rankings[retrieved_classes] = range(num_classes)

    # Which of these is a true label?

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)

    retrieved_class_true[class_rankings[pos_class_indices]] = True

    # Num hits for every truncated retrieval list.

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    # Precision of retrieval list truncated at each hit, in order of pos_labels.

    precision_at_hits = (

            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /

            (1 + class_rankings[pos_class_indices].astype(np.float)))

    return pos_class_indices, precision_at_hits





def calculate_per_class_lwlrap(truth, scores):

    """Calculate label-weighted label-ranking average precision.



    Arguments:

      truth: np.array of (num_samples, num_classes) giving boolean ground-truth

        of presence of that class in that sample.

      scores: np.array of (num_samples, num_classes) giving the classifier-under-

        test's real-valued score for each class for each sample.



    Returns:

      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each

        class.

      weight_per_class: np.array of (num_classes,) giving the prior of each

        class within the truth labels.  Then the overall unbalanced lwlrap is

        simply np.sum(per_class_lwlrap * weight_per_class)

    """

    assert truth.shape == scores.shape

    num_samples, num_classes = scores.shape

    # Space to store a distinct precision value for each class on each sample.

    # Only the classes that are true for each sample will be filled in.

    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):

        pos_class_indices, precision_at_hits = (

            _one_sample_positive_class_precisions(scores[sample_num, :],

                                                  truth[sample_num, :]))

        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

            precision_at_hits)

    labels_per_class = np.sum(truth > 0, axis=0)

    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    # Form average of each column, i.e. all the precisions assigned to labels in

    # a particular class.

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /

                        np.maximum(1, labels_per_class))

    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions

    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)

    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples

    #                = np.sum(per_class_lwlrap * weight_per_class)

    return per_class_lwlrap, weight_per_class





# Accumulator object version.



class lwlrap_accumulator(object):

  """Accumulate batches of test samples into per-class and overall lwlrap."""  



  def __init__(self):

    self.num_classes = 0

    self.total_num_samples = 0

  

  def accumulate_samples(self, batch_truth, batch_scores):

    """Cumulate a new batch of samples into the metric.

    

    Args:

      truth: np.array of (num_samples, num_classes) giving boolean

        ground-truth of presence of that class in that sample for this batch.

      scores: np.array of (num_samples, num_classes) giving the 

        classifier-under-test's real-valued score for each class for each

        sample.

    """

    assert batch_scores.shape == batch_truth.shape

    num_samples, num_classes = batch_truth.shape

    if not self.num_classes:

      self.num_classes = num_classes

      self._per_class_cumulative_precision = np.zeros(self.num_classes)

      self._per_class_cumulative_count = np.zeros(self.num_classes, 

                                                  dtype=np.int)

    assert num_classes == self.num_classes

    for truth, scores in zip(batch_truth, batch_scores):

      pos_class_indices, precision_at_hits = (

        _one_sample_positive_class_precisions(scores, truth))

      self._per_class_cumulative_precision[pos_class_indices] += (

        precision_at_hits)

      self._per_class_cumulative_count[pos_class_indices] += 1

    self.total_num_samples += num_samples



  def per_class_lwlrap(self):

    """Return a vector of the per-class lwlraps for the accumulated samples."""

    return (self._per_class_cumulative_precision / 

            np.maximum(1, self._per_class_cumulative_count))



  def per_class_weight(self):

    """Return a normalized weight vector for the contributions of each class."""

    return (self._per_class_cumulative_count / 

            float(np.sum(self._per_class_cumulative_count)))



  def overall_lwlrap(self):

    """Return the scalar overall lwlrap for cumulated samples."""

    return np.sum(self.per_class_lwlrap() * self.per_class_weight())
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, 3, 1, 1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

        )



        self._init_weights()

        

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.zeros_(m.bias)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = F.avg_pool2d(x, 2)

        return x

    

class Classifier(nn.Module):

    def __init__(self, num_classes=1000): # <======== modificaition to comply fast.ai

        super().__init__()

        

        self.conv = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=64),

            ConvBlock(in_channels=64, out_channels=128),

            ConvBlock(in_channels=128, out_channels=256),

            ConvBlock(in_channels=256, out_channels=512),

        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # <======== modificaition to comply fast.ai

        self.fc = nn.Sequential(

            nn.Dropout(0.2),

            nn.Linear(512, 128),

            nn.PReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.1),

            nn.Linear(128, num_classes),

        )



    def forward(self, x):

        x = self.conv(x)

        #x = torch.mean(x, dim=3)   # <======== modificaition to comply fast.ai

        #x, _ = torch.max(x, dim=2) # <======== modificaition to comply fast.ai

        x = self.avgpool(x)         # <======== modificaition to comply fast.ai

        x = self.fc(x)

        return x
# !!! use globals CUR_X_FILES, CUR_X

def open_fat2019_image(fn, convert_mode, after_open)->Image:

    # open

    fname = fn.split('/')[-1]

    if '!' in fname:

        fname, crop_x = fname.split('!')

        crop_x = int(crop_x)

    else:

        crop_x = -1

    idx = CUR_X_FILES.index(fname)

    x = CUR_X[idx]

    # crop

    base_dim, time_dim, _ = x.shape

    if crop_x == -1:

        crop_x = random.randint(0, time_dim - base_dim)

    x = x[0:base_dim, crop_x:crop_x+base_dim, :]

    x = np.transpose(x, (1, 0, 2))

    x = np.transpose(x, (2, 1, 0))

    # standardize

    return Image(torch.from_numpy(x.astype(np.float32, copy=False)).div_(255))





vision.data.open_image = open_fat2019_image
CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test
output = StringIO()

csv_writer = writer(output)

csv_writer.writerow(test_df.columns)



for _, row in tqdm_notebook(test_df.iterrows(), total=test_df.shape[0]):

    idx = CUR_X_FILES.index(row.fname)

    time_dim = CUR_X[idx].shape[1]

    s = math.ceil((time_dim-conf.n_mels) / PREDICTION_WINDOW_SHIFT) + 1

    

    fname = row.fname

    for crop_x in [int(np.around((time_dim-conf.n_mels)*x/(s-1))) if s != 1 else 0 for x in range(s)]:

        row.fname = fname + '!' + str(crop_x)

        csv_writer.writerow(row)



output.seek(0)

test_df_multi = pd.read_csv(output)



del row, test_df, output, csv_writer; gc.collect();
test = ImageList.from_df(test_df_multi, models_list[0][0])



for model_nb, (work, name) in enumerate(models_list):

    for fold in range(n_splits):

        learn = load_learner(work, name.format(fold=fold), test=test)

        preds, _ = learn.get_preds(ds_type=DatasetType.Test)

        preds = preds.cpu().numpy()

        if (fold == 0) and (model_nb == 0):

            predictions = preds

        else:

            predictions += preds



predictions /= (n_splits * len(models_list))
test_df_multi[learn.data.classes] = predictions

test_df_multi['fname'] = test_df_multi.fname.apply(lambda x: x.split('!')[0])
submission = test_df_multi.infer_objects().groupby('fname').mean().reset_index()

submission.to_csv('submission.csv', index=False)

submission.head()
submission.set_index('fname').idxmax(1)