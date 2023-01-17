import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import torch.optim as optim

from torchvision import datasets, transforms

from types import SimpleNamespace

import matplotlib.pyplot as plt

import csv

import librosa

import scipy as sc

from fastai import *
import fastai

fastai.__version__
from fastai.metrics import accuracy

from fastai.torch_core import *





from fastai.vision import models, ClassificationInterpretation
# From here I start importing FastAi Audio from: https://github.com/sevenfx/fastai_audio 

# I had to do it this way because it didn't work propperly in any other way

# It is a kind of buggy, beta Version so it took me quiet some time to debug it

# It is really fine to train a model with but the Testset doesn't really work

# Here is the newest version: https://github.com/mogwai/fastai_audio
from fastai.torch_core import *

from scipy.io import wavfile

from IPython.display import display, Audio



__all__ = ['AudioClip', 'open_audio']





class AudioClip(ItemBase):

    def __init__(self, signal, sample_rate):

        self.data = signal

        self.sample_rate = sample_rate



    def __str__(self):

        return '(duration={}s, sample_rate={:.1f}KHz)'.format(

            self.duration, self.sample_rate/1000)



    def clone(self):

        return self.__class__(self.data.clone(), self.sample_rate)



    def apply_tfms(self, tfms, **kwargs):

        x = self.clone()

        for tfm in tfms:

            x.data = tfm(x.data)

        return x



    @property

    def num_samples(self):

        return len(self.data)



    @property

    def duration(self):

        return self.num_samples / self.sample_rate



    def show(self, ax=None, figsize=(5, 1), player=True, title=None, **kwargs):

        if ax is None:

            _, ax = plt.subplots(figsize=figsize)

        if title:

            ax.set_title(title)

        timesteps = np.arange(len(self.data)) / self.sample_rate

        ax.plot(timesteps, self.data)

        ax.set_xlabel('Time (s)')

        plt.show()

        if player:

            # unable to display an IPython 'Audio' player in plt axes

            display(Audio(self.data, rate=self.sample_rate))





def open_audio(fn):

    sr, x = wavfile.read(fn)

    t = torch.from_numpy(x.astype(np.float32, copy=False))

    if x.dtype == np.int16:

        t.div_(32767)

    elif x.dtype != np.float32:

        raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))

    return AudioClip(t, sr)

from fastai.torch_core import *

from fastai.train import Learner



from fastai.callbacks.hooks import num_features_model

from fastai.vision import create_body, create_head

from fastai.vision.learner import cnn_config, _resnet_split



__all__ = ['create_cnn']





# copied from fastai.vision.learner, omitting unused args,

# and adding channel summing of first convolutional layer

def create_cnn(data, arch, pretrained=True, sum_channel_weights=True, **kwargs):

    meta = cnn_config(arch)

    body = create_body(arch, pretrained)



    # sum up the weights of in_channels axis, to reduce to single input channel

    # Suggestion by David Gutman

    # https://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/2

    if sum_channel_weights:

        first_conv_layer = body[0]

        first_conv_weights = first_conv_layer.state_dict()['weight']

        assert first_conv_weights.size(1) == 3 # RGB channels dim

        summed_weights = torch.sum(first_conv_weights, dim=1, keepdim=True)

        first_conv_layer.weight.data = summed_weights

        first_conv_layer.in_channels = 1



    nf = num_features_model(body) * 2

    head = create_head(nf, data.c, None, 0.5)

    model = nn.Sequential(body, head)

    learn = Learner(data, model, **kwargs)

    learn.split(meta['split'])

    if pretrained:

        learn.freeze()

    apply_init(model[1], nn.init.kaiming_normal_)

    return learn

from fastai.torch_core import *



__all__ = ['mapk']





def mapk_np(preds, targs, k=3):

    preds = np.argsort(-preds, axis=1)[:, :k]

    score = 0.

    for i in range(k):

        num_hits = (preds[:, i] == targs).sum()

        score += num_hits * (1. / (i+1.))

    score /= preds.shape[0]

    return score





def mapk(preds, targs, k=3):

    return tensor(mapk_np(to_np(preds), to_np(targs), k))

from fastai.basic_data import *

from fastai.data_block import *

from fastai.data_block import _maybe_squeeze

from fastai.text import SortSampler, SortishSampler

from fastai.torch_core import *











def pad_collate1d(batch):

    xs, ys = zip(*to_data(batch))

    max_len = max(x.size(0) for x in xs)

    padded_xs = torch.zeros(len(xs), max_len, dtype=xs[0].dtype)

    for i,x in enumerate(xs):

        padded_xs[i,:x.size(0)] = x

    return padded_xs, tensor(ys)





# TODO: generalize this away from hard coding dim values

def pad_collate2d(batch):

    xs, ys = zip(*to_data(batch))

    max_len = max(max(x.size(1) for x in xs), 1)

    bins = xs[0].size(0)

    padded_xs = torch.zeros(len(xs), bins, max_len, dtype=xs[0].dtype)

    for i,x in enumerate(xs):

        padded_xs[i,:,:x.size(1)] = x

    return padded_xs, tensor(ys)





class AudioDataBunch(DataBunch):

    @classmethod

    def create(cls, train_ds, valid_ds, test_ds=None, path='.',

               bs=64, equal_lengths=True, length_col=None, dl_tfms=None):

        if equal_lengths:

            return super().create(train_ds, valid_ds, test_ds=test_ds, path=path,

                                  bs=bs, dl_tfms=dl_tfms)

        else:

            datasets = super()._init_ds(train_ds, valid_ds, test_ds)

            train_ds, valid_ds, fix_ds = datasets[:3]

            if len(datasets) == 4:

                test_ds = datasets[3]



            train_lengths = train_ds.lengths(length_col)

            train_sampler = SortishSampler(train_ds.x, key=lambda i: train_lengths[i], bs=bs//2)

            train_dl = DataLoader(train_ds, batch_size=bs, sampler=train_sampler)



            # precalculate lengths ahead of time if they aren't included in xtra

            valid_lengths = valid_ds.lengths(length_col)

            valid_sampler = SortSampler(valid_ds.x, key=lambda i: valid_lengths[i])

            valid_dl = DataLoader(valid_ds, batch_size=bs, sampler=valid_sampler)



            fix_lengths = fix_ds.lengths(length_col)

            fix_sampler = SortSampler(fix_ds.x, key=lambda i: fix_lengths[i])

            fix_dl = DataLoader(fix_ds, batch_size=bs, sampler=fix_sampler)



            dataloaders = [train_dl, valid_dl, fix_dl]

            if test_ds is not None:

                test_dl = DataLoader(test_dl, batch_size=1)

                dataloaders.append(test_dl)



            return cls(*dataloaders, path=path, collate_fn=pad_collate1d, tfms=tfms)



    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train):

        dl = self.dl(ds_type)

        ds = dl.dl.dataset

        idx = np.random.choice(len(ds), size=rows, replace=False)

        batch = ds[idx]

        xs, ys = batch.x, batch.y

        self.train_ds.show_xys(xs, ys)





class AudioItemList(ItemList):

    """NOTE: this class has been heavily adapted from ImageItemList"""

    _bunch = AudioDataBunch



    @classmethod

    def open(cls, fn):

        return open_audio(fn)



    def get(self, i):

        fn = super().get(i)

        return self.open(fn)



    def lengths(self, length_col=None):

        if length_col is not None and self.xtra is not None:

            lengths = self.xtra.iloc[:, df_names_to_idx(length_col, self.xtra)]

            lengths = _maybe_squeeze(lengths.values)

        else:

            lengths = [clip.num_samples for clip in self]

        return lengths



    @classmethod

    def from_df(cls, df, path, col=0, folder='.', suffix='', length_col=None):

        """Get the filenames in `col` of `df` and will had `path/folder` in front of them,

        `suffix` at the end. `create_func` is used to open the audio files."""

        suffix = suffix or ''

        res = super().from_df(df, path=path, col=col, length_col=length_col)

        res.items = np.char.add(np.char.add(f'{folder}/', res.items.astype(str)), suffix)

        res.items = np.char.add(f'{res.path}/', res.items)

        return res



    def show_xys(self, xs, ys, figsize=None, **kwargs):

        for x, y in zip(xs, ys):

            x.show(title=y, **kwargs)



import librosa as lr

from fastai.torch_core import *



__all__ = ['get_frequency_transforms', 'get_frequency_batch_transforms',

           'FrequencyToMel', 'ToDecibels', 'Spectrogram']





def get_frequency_transforms(n_fft=2048, n_hop=512, window=torch.hann_window,

                             n_mels=None, f_min=0, f_max=None, sample_rate=44100,

                             decibels=True, ref='max', top_db=80.0, norm_db=True):

    tfms = [Spectrogram(n_fft=n_fft, n_hop=n_hop, window=window)]

    if n_mels is not None:

        tfms.append(FrequencyToMel(n_mels=n_mels, n_fft=n_fft, sr=sample_rate,

                                   f_min=f_min, f_max=f_max))

    if decibels:

        tfms.append(ToDecibels(ref=ref, top_db=top_db, normalized=norm_db))



    # only one list, as its applied to all dataloaders

    return tfms





def get_frequency_batch_transforms(*args, add_channel_dim=True, **kwargs):

    tfms = get_frequency_transforms(*args, **kwargs)



    def _freq_batch_transformer(inputs):

        xs, ys = inputs

        for tfm in tfms:

            xs = tfm(xs)

        if add_channel_dim:

            xs.unsqueeze_(1)

        return xs, ys

    return [_freq_batch_transformer]





class FrequencyToMel:

    def __init__(self, n_mels=40, n_fft=1024, sr=16000,

                 f_min=0.0, f_max=None, device=None):

        mel_fb = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,

                                fmin=f_min, fmax=f_max).astype(np.float32)

        self.mel_filterbank = to_device(torch.from_numpy(mel_fb), device)



    def __call__(self, spec_f):

        spec_m = self.mel_filterbank @ spec_f

        return spec_m





class ToDecibels:

    def __init__(self,

                 power=2, # magnitude=1, power=2

                 ref=1.0,

                 top_db=None,

                 normalized=True,

                 amin=1e-7):

        self.constant = 10.0 if power == 2 else 20.0

        self.ref = ref

        self.top_db = abs(top_db) if top_db else top_db

        self.normalized = normalized

        self.amin = amin



    def __call__(self, x):

        batch_size = x.shape[0]

        if self.ref == 'max':

            ref_value = x.contiguous().view(batch_size, -1).max(dim=-1)[0]

            ref_value.unsqueeze_(1).unsqueeze_(1)

        else:

            ref_value = tensor(self.ref)

        spec_db = x.clamp_min(self.amin).log10_().mul_(self.constant)

        spec_db.sub_(ref_value.clamp_min_(self.amin).log10_().mul_(10.0))

        if self.top_db is not None:

            max_spec = spec_db.view(batch_size, -1).max(dim=-1)[0]

            max_spec.unsqueeze_(1).unsqueeze_(1)

            spec_db = torch.max(spec_db, max_spec - self.top_db)

            if self.normalized:

                # normalize to [0, 1]

                spec_db.add_(self.top_db).div_(self.top_db)

        return spec_db





# Returns power spectrogram (magnitude squared)

class Spectrogram:

    def __init__(self, n_fft=1024, n_hop=256, window=torch.hann_window,

                 device=None):

        self.n_fft = n_fft

        self.n_hop = n_hop

        self.window = to_device(window(n_fft), device)



    def __call__(self, x):

        X = torch.stft(x,

                       n_fft=self.n_fft,

                       hop_length=self.n_hop,

                       win_length=self.n_fft,

                       window=self.window,

                       onesided=True,

                       center=True,

                       pad_mode='constant',

                       normalized=True)

        # compute power from real and imag parts (magnitude^2)

        X.pow_(2.0)

        power = X[:,:,:,0] + X[:,:,:,1]

        return power

"Brings TTA (Test Time Functionality) to the `Learner` class. Use `learner.TTA()` instead"

from fastai.torch_core import *

from fastai.basic_train import *

from fastai.basic_train import _loss_func2activ

from fastai.basic_data import DatasetType



__all__ = []





def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid) -> Iterator[List[Tensor]]:

    "Computes the outputs for several augmented inputs for TTA"

    dl = learn.dl(ds_type)

    ds = dl.dataset

    old = ds.tfms

    augm_tfm = [o for o in learn.data.train_ds.tfms]

    try:

        pbar = master_bar(range(8))

        for i in pbar:

            ds.tfms = augm_tfm

            yield get_preds(learn.model, dl, pbar=pbar, activ=_loss_func2activ(learn.loss_func))[0]

    finally: ds.tfms = old





Learner.tta_only = _tta_only





def _TTA(learn:Learner, beta:float=0.4, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False) -> Tensors:

    "Applies TTA to predict on `ds_type` dataset."

    preds,y = learn.get_preds(ds_type)

    all_preds = list(learn.tta_only(ds_type=ds_type))

    avg_preds = torch.stack(all_preds).mean(0)

    if beta is None: return preds,avg_preds,y

    else:

        final_preds = preds*beta + avg_preds*(1-beta)

        if with_loss:

            return final_preds, y, calc_loss(final_preds, y, learn.loss_func)

        return final_preds, y





Learner.TTA = _TTA

# Here the importing Stops
# Hyperparameters

use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')
AUDIO = "../input/oeawai/train/kaggle-train"

test_ds = "../input/oeawai/kaggle-test/kaggle-test"
n_fft = 1024 # output of fft will have shape [513 x n_frames]

n_hop = 256  # 75% overlap between frames

n_mels = 40 # compress 513 dimensions to 40 via mel frequency scale

sample_rate = 16000



dl_tfms = get_frequency_batch_transforms(n_fft=n_fft, n_hop=n_hop,

                                      n_mels=n_mels, sample_rate=sample_rate)
np.random.seed(42)
batch_size = 64

# works for training without the test_ds=test_ds

instrument_family_pattern = r'(\w+)_\w+_\d+-\d+-\d+.wav$'

data = (AudioItemList

            .from_folder(AUDIO)

            .split_by_rand_pct(0.2)

            .label_from_re(instrument_family_pattern)

            .databunch(bs=batch_size, dl_tfms=dl_tfms))
#.split_by_rand_pct(0.2)
data.c, data.classes

learn = create_cnn(data, models.resnet18, metrics=accuracy)

learn.model_dir='/kaggle/working/'
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(3, 1.32E-02)
#learn.save('resnet18_fastai_audio_3')
learn.export('/kaggle/working/resnet18_fastai_audio_2_trained_model_3epochs.pkl')
# Testing was done locally due to some problems with the transformations so the whole test set was transformed to pictures and then tested.

# This 
learn= load_learner('/kaggle/working/','resnet18_fastai_audio_2_trained_model.pkl')