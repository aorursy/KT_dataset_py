import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



import fastai

from fastai.imports import *

from fastai.basics import *

from fastai.metrics import accuracy

from fastai.torch_core import *

from fastai.vision import *

fastai.__version__

import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import IPython

import IPython.display

import PIL

import pickle

import time

start_time = time.time()
from fastai.torch_core import *

import librosa

from scipy.io import wavfile

from IPython.display import display, Audio



import importlib

soundfile_spec = importlib.util.find_spec("soundfile")

if soundfile_spec is not None:

    import soundfile as sf



class AudioClip(ItemBase):

    def __init__(self, signal, sample_rate, fn):

        self.data = signal # Contains original signal to start 

        self.original_signal = signal.clone()

        self.processed_signal = signal.clone()

        self.sample_rate = sample_rate

        self.fn = fn



    def __str__(self):

        return '(duration={}s, sample_rate={:.1f}KHz)'.format(

            self.duration, self.sample_rate/1000)



    def clone(self):

        return self.__class__(self.data.clone(), self.sample_rate, self.fn)



    def apply_tfms(self, tfms, **kwargs):

        for tfm in tfms:

            self.data = tfm(self.data, **kwargs)

            if issubclass(type(tfm), MyDataAugmentation):

                self.processed_signal = self.data.clone().cpu()

        return self

    

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

            ax.set_title("Class: " + str(title) + " \nfilename: " + str(self.fn))

        

        timesteps = np.arange(self.original_signal.shape[1]) / self.sample_rate

        

        ax.plot(timesteps, self.original_signal[0]) 

        if self.original_signal.size(0) > 1: # Check if mono or stereo

            ax.plot(timesteps, self.original_signal[1]) 

        ax.set_xlabel('Original Signal Time (s)')

        plt.show()

        

        timesteps = np.arange(self.processed_signal.shape[1]) / self.sample_rate



        _, ax = plt.subplots(figsize=figsize)

        if title:

            ax.set_title("Class: " + str(title) + " \nfilename: " + str(self.fn))

        ax.plot(timesteps, self.processed_signal[0]) 

        if self.processed_signal.size(0) > 1: # Check if mono or stereo

            ax.plot(timesteps, self.processed_signal[1]) 

        ax.set_xlabel('Processed Signal Time (s)')

        plt.show()

        

        if player:

            # unable to display an IPython 'Audio' player in plt axes

            display("Original signal")

            display(Audio(self.original_signal, rate=self.sample_rate))

            display("Processed signal")

            display(Audio(self.processed_signal, rate=self.sample_rate))



def open_audio(fn, using_librosa:bool=False, downsampling=8000):

    if using_librosa: 

        x, sr = librosa.core.load(fn, sr=None, mono=False)

        

    else:

        if soundfile_spec is not None:

            x, sr = sf.read(fn, always_2d=True, dtype="float32")

        else:

            raise Exception("Cannot load soundfile")

            #sr, x = wavfile.read(fn) # 10 times faster than librosa but issues with 24bits wave

    

    if len(x.shape) == 1: # Mono signal

        x = x.reshape(1, -1) # Add 1 channel

    else:

        if not using_librosa:

            x = np.swapaxes(x, 1, 0) # Scipy result is timestep * channels instead of channels * timestep

    

    if downsampling is not None:

        x = librosa.core.resample(x, sr, downsampling)

        sr = downsampling

    t = torch.from_numpy(x.astype(np.float32, copy=False))

    if x.dtype == np.int16:

        t.div_(32767)

    elif x.dtype != np.float32:

        raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))

    return AudioClip(t, sr, fn)
from fastai.basic_data import *

from fastai.data_block import *

from fastai.data_block import _maybe_squeeze

from fastai.text import SortSampler, SortishSampler

from fastai.vision import *

from fastai.torch_core import *

import gc

import copy





class AudioDataBunch(DataBunch):

    

    # Subclass because of bug to give dl_tfms to underlying dataloader

    @classmethod

    def create(cls, train_ds, valid_ds, 

               tfms:Optional[Collection[Callable]]=None, # There is a bug in LabelLists because dl_tfms is not given to dataloader

               **kwargs)->'AudioDataBunch':

        db = super().create(train_ds=train_ds, valid_ds=valid_ds, dl_tfms=tfms, **kwargs)



        return db







    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, **kwargs):

        dl = self.dl(ds_type)

        ds = dl.dl.dataset



        idx = np.random.choice(len(ds), size=rows, replace=False)

        batch = ds[idx]

        

        max_count = min(rows, len(batch))

        xs, ys, xs_processed, ys_processed = [], [], [], []

        for i in range(max_count):

            x, x_processed, y, y_processed = batch[i][0], batch[i][0].data, batch[i][1], torch.tensor(batch[i][1].data)

            xs.append(x)

            xs_processed.append(x_processed)

            ys.append(y)

            ys_processed.append(y_processed)



        xs_processed = torch.stack(xs_processed, dim=0)

        ys_processed = torch.stack(ys_processed, dim=0)

        

        for tfm in dl.tfms:

            xs_processed, ys_processed = tfm((xs_processed, ys_processed))



        

        self.train_ds.show_xys(xs, ys, xs_processed=xs_processed.unbind(dim=0), **kwargs)

        del xs, ys, xs_processed, ys_processed



    # Inspired by ImageDataBunch

    def batch_stats(self, funcs:Collection[Callable]=None, ds_type:DatasetType=DatasetType.Train)->Tensor:

        "Grab a batch of data and call reduction function `func` per channel"

        funcs = ifnone(funcs, [torch.mean,torch.std])

        x = self.one_batch(ds_type=ds_type, denorm=False)[0].cpu()

        return [func(channel_view(x), 1) for func in funcs]

        

    # Inspired by ImageDataBunch

    def normalize(self, stats:Collection[Tensor]=None, do_x:bool=True, do_y:bool=False)->None:

        "Add normalize transform using `stats` (defaults to `DataBunch.batch_stats`)"

        if getattr(self,'norm',False): raise Exception('Can not call normalize twice')

        if stats is None: self.stats = self.batch_stats()

        else:             self.stats = stats

        self.norm,self.denorm = normalize_funcs(*self.stats, do_x=do_x, do_y=do_y)

        self.add_tfm(self.norm)

        return self



       



# Inspired by https://docs.fast.ai/tutorial.itemlist.html

class AudioItemList(ItemList):

    _bunch = AudioDataBunch # Needed to include normalize

    

    def __init__(self, items:Iterator,

                 using_librosa=False, downsampling=None, **kwargs):

        super().__init__(items=items, **kwargs)

        self.using_librosa = using_librosa

        self.copy_new.append('using_librosa')

        self.downsampling = downsampling

        self.copy_new.append('downsampling')



    def get(self, i):

        fn = super().get(i)

        return open_audio(self.path/fn, using_librosa=self.using_librosa, downsampling=self.downsampling)





    @classmethod

    def from_df(cls, df, path, using_librosa=False, folder:PathOrStr=None, downsampling=None, **kwargs):

        #if folder is not None: path2 = str(path)+folder

        res = super().from_df(df, path=path, **kwargs)

        pref = f'{res.path}{os.path.sep}'

        if folder is not None: pref += f'{folder}{os.path.sep}'

        res.items = np.char.add(pref, res.items.astype(str))

        res.using_librosa=using_librosa

        res.downsampling = downsampling

        return res

    

    

    def reconstruct(self, t:Tensor, x:Tensor = None): 

        raise Exception("Not implemented yet")

        # From torch

        #return ImagePoints(FlowField(x.size, t), scale=False)



    

    

    def show_xys(self, xs, ys, xs_processed=None, figsize=None, **kwargs):

        if xs_processed is None:

            for x, y in zip(xs, ys):

                x.show(title=str(y), figsize=figsize, **kwargs)

        else:

            for x, y, x_processed in zip(xs, ys, xs_processed):

                x.show(title=str(y), figsize=figsize, **kwargs)

                for channel in range(x_processed.size(0)):

                    Image(x_processed[channel, :, :].unsqueeze(0)).show(figsize=figsize)
from fastai.torch_core import *

from fastai.train import Learner



from fastai.callbacks.hooks import num_features_model, hook_output

from fastai.vision import create_body, create_head, Image

from fastai.vision.learner import cnn_config, _resnet_split, ClassificationInterpretation









# copied from fastai.vision.learner, omitting unused args,

# and adding channel summing of first convolutional layer

def create_cnn(data, arch, pretrained=False, is_mono_input=True, **kwargs):

    meta = cnn_config(arch)

    body = create_body(arch, pretrained)



    # sum up the weights of in_channels axis, to reduce to single input channel

    # Suggestion by David Gutman

    # https://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/2

    if is_mono_input:

        first_conv_layer = body[0][0]

        first_conv_weights = first_conv_layer.state_dict()['weight']

        assert first_conv_weights.size(1) == 3 # RGB channels dim

        summed_weights = torch.sum(first_conv_weights, dim=1, keepdim=True)

        first_conv_layer.weight.data = summed_weights

        first_conv_layer.in_channels = 1

    else:

        # In this case, the input is a stereo

        first_conv_layer = body[0]

        first_conv_weights = first_conv_layer.state_dict()['weight']

        assert first_conv_weights.size(1) == 3 # RGB channels dim

        summed_weights = torch.sum(first_conv_weights, dim=1, keepdim=True)

        first_conv_layer.weight.data = first_conv_weights[:, :2, :, :] # Keep only 2 channels for the weights

        first_conv_layer.in_channels = 2



    nf = num_features_model(body) * 2

    head = create_head(nf, data.c, None, 0.5)

    model = nn.Sequential(body, head)

    learn = Learner(data, model, **kwargs)

    learn.split(meta['split'])

    if pretrained:

        learn.freeze()

    apply_init(model[1], nn.init.kaiming_normal_)

    return learn







def my_cl_int_plot_top_losses(self, k, largest=True, figsize=(25,7), heatmap:bool=True, heatmap_thresh:int=16,

                            return_fig:bool=None)->Optional[plt.Figure]:

    "Show images in `top_losses` along with their prediction, actual, loss, and probability of actual class."

    tl_val,tl_idx = self.top_losses(k, largest)

    classes = self.data.classes

    cols = math.ceil(math.sqrt(k))

    rows = math.ceil(k/cols)

    fig,axes = plt.subplots(rows, cols, figsize=figsize)

    fig.suptitle('prediction/actual/loss/probability', weight='bold', size=14)

    for i,idx in enumerate(tl_idx):

        audio, cl = self.data.dl(self.ds_type).dataset[idx]

        audio = audio.clone()

        

        m = self.learn.model.eval()

        

        x, _ = self.data.one_item(audio) # Process one audio into prediction

        

        x_consolidated = x.sum(dim=1, keepdim=True) # Sum accross all channels to ease the interpretation



        im = Image(x_consolidated[0, :, :, :].cpu()) # Extract the processed image from the prediction (after dl_tfms) and keep it into CPU

        cl = int(cl)

        title = f'{classes[self.pred_class[idx]]}/{classes[cl]} / {self.losses[idx]:.2f} / {self.probs[idx][cl]:.2f}'

        title = title + f'\n {audio.fn}'

        

        im.show(ax=axes.flat[i], title=title)

        

        if heatmap:

            # Related paper http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf

            with hook_output(m[0]) as hook_a: # hook activations from CNN module

                with hook_output(m[0], grad= True) as hook_g: # hook gradients from CNN module

                    preds = m(x) # Forward pass to get activations

                    preds[0,cl].backward() # Backward pass to get gradients

            acts = hook_a.stored[0].cpu()

            if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:

                grad = hook_g.stored[0][0].cpu() # Hook the gradients from the CNN module and extract the first one (because one item only)

                grad_chan = grad.mean(1).mean(1) # Mean accross image to keep mean gradients per channel 

                mult = F.relu(((acts*grad_chan[...,None,None])).sum(0)) # Multiply activation with gradients (add 1 dim for height and width)

                sz = list(im.shape[-2:])

                axes.flat[i].imshow(mult, alpha=0.35, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap='magma')     

        

    if ifnone(return_fig, defaults.return_fig): return fig

    

    

ClassificationInterpretation.plot_audio_top_losses = my_cl_int_plot_top_losses
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
import librosa as lr

from fastai.torch_core import *

import gc





def get_data_augmentation_transforms(max_seconds=30, start_at_second=0,

                                     sample_rate=44100, noise_scl=None, convert_to_mono=True):

    tfms = []

    if convert_to_mono:

        tfms.append(ConvertToMono())

    max_channels = 1 if convert_to_mono else 2

    tfms.append(PadToMax(start_at_second=start_at_second, max_seconds=max_seconds, 

                         sample_rate=sample_rate, max_channels=max_channels))

    

    if noise_scl is not None:

        tfms.append(WhiteNoise(noise_scl))

    return tfms



def get_frequency_transforms(n_fft=512, n_hop=160, top_db=80,

                             n_mels=None, f_min=0, f_max=None, sample_rate=44100):

#    tfms.append(MFCC(n_fft=n_fft, n_mfcc=n_mels, hop_length=n_hop, sample_rate=sample_rate, f_min=f_min, f_max=f_max))

    tfms = [Spectrogram(n_fft=n_fft, n_hop=n_hop)]

    tfms.append(FrequencyToMel(n_fft=n_fft, n_mels=n_mels, sr=sample_rate, f_min=f_min, f_max=f_max))

    tfms.append(ToDecibels(top_db=top_db))

    

    return tfms





def get_frequency_batch_transforms(*args, **kwargs):

    tfms = get_frequency_transforms(*args, **kwargs)



    def _freq_batch_transformer(inputs):

        xs, ys = inputs

        for tfm in tfms:

            xs = tfm(xs)

        del inputs

        

        return xs, ys.detach()

    return [_freq_batch_transformer]



# Parent classes used to distinguish transforms for data augmentation and transforms to convert audio into image

class MyDataAugmentation:

    pass



class MySoundToImage:

    pass



### The below transformers are on the single AudioClip (to help to keep tracks of changes from data augmentation)



class ConvertToMono(MyDataAugmentation):

    def __init__(self):

        pass



    def __call__(self, X):

        assert(X.dim() == 2) # channels * timestep

        X = X.sum(0) # Sum over channels

        X = X.unsqueeze(0)

        assert(X.dim() == 2) # channels * timestep

        return X



    

    

class PadToMax(MyDataAugmentation):

    def __init__(self, start_at_second=0, max_seconds=30, sample_rate=16000, max_channels=1):

        self.max_seconds = max_seconds

        self.sample_rate = sample_rate

        self.max_channels = max_channels

        self.start_at_second = start_at_second

        



    def __call__(self, X):

        # X must be channels * timestep

        assert(X.dim() == 2)

        assert(X.size(0) <= 2) # There is only 2 channels at maximum 

        

        mx = int(self.max_seconds * self.sample_rate)

        start_at = min(int(self.start_at_second * self.sample_rate), X.size(1))

        if X.size(1) - start_at <= mx:

            start_at = max(X.size(1) - mx, 0)

        

        if (X.size(1) < mx): 

            X = torch.cat((X, torch.zeros([X.size(0), mx - X.size(1)], device=X.device)), dim=1) # Channels * Timestep

        if (X.size(1) > mx): 

            X = X[:, start_at:(mx + start_at)]

        if X.size(0) < self.max_channels:

            targets = torch.zeros(self.max_channels, X.size(1), device=X.device)

            targets[:X.size(0), :] = X

            X = targets

        

        return X



    

class WhiteNoise(MyDataAugmentation):

    def __init__(self, noise_scl=0.0005):

        self.noise_scl= noise_scl



    def __call__(self, X):

        noise = torch.randn(X.shape, device=X.device) * self.noise_scl 

        assert(X.dim() == 2) # channels * timestep

        return X + noise



    

### The below transformers are on the whole batch



    

class MFCCLibrosa(MySoundToImage):

    def __init__(self, sample_rate=16000, n_mfcc=20, n_fft=512, hop_length=512, f_min=0, f_max=None):

        self.n_mfcc = n_mfcc

        self.sample_rate = sample_rate

        self.n_fft = n_fft

        self.hop_length = hop_length

        self.f_min=f_min

        self.f_max=f_max

    

    def __call__(self, X):

        mfcc = torch.zeros([X.size(0), self.n_mfcc, 1+int(X.size(1) / self.hop_length)], device=X.device)

        for i in range(X.size(0)):

            single_mfcc = lr.feature.mfcc(y=X[0, :].cpu().numpy(), 

                                   sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length,

                                         fmin=self.f_min, fmax=self.f_max)

            mfcc[i, :, :] = torch.tensor(single_mfcc, device=X.device)

        del X

        return mfcc

    

# Returns power spectrogram (magnitude squared)

class Spectrogram(MySoundToImage):

    def __init__(self, n_fft=1024, n_hop=256, window=torch.hann_window,

                 device=None):

        self.n_fft = n_fft

        self.n_hop = n_hop

        self.window = window(n_fft)



    def __call__(self, x):

        X_left = torch.stft(x[:, 0, :],

                       n_fft=self.n_fft,

                       hop_length=self.n_hop,

                       win_length=self.n_fft,

                       window=to_device(self.window, x.device),

                       onesided=True,

                       center=True,

                       pad_mode='constant',

                       normalized=True)

        # compute power from real and imag parts (magnitude^2)

        X_left.pow_(2.0)

        X_left = X_left[:,:,:,0] + X_left[:,:,:,1]

        X_left = X_left.unsqueeze(1) # Add channel dimension



        if (x.size(1) > 1):

            X_right = torch.stft(x[:, 1, :],

                           n_fft=self.n_fft,

                           hop_length=self.n_hop,

                           win_length=self.n_fft,

                           window=to_device(self.window, x.device),

                           onesided=True,

                           center=True,

                           pad_mode='constant',

                           normalized=True)        

            # compute power from real and imag parts (magnitude^2)

            X_right.pow_(2.0)

            X_right = X_right[:,:,:,0] + X_right[:,:,:,1]

            X_right = X_right.unsqueeze(1) # Add channel dimension

            res = torch.cat([X_left, X_right], dim=1) 

            assert(res.dim() == 4) # Check dim (n sample * channels * h * w)

            return res

            

        else:

            assert(X_left.dim() == 4) # Check dim (n sample * channels * h * w)

            return X_left # Return only mono channel

        

    

class FrequencyToMel(MySoundToImage):

    def __init__(self, n_mels=40, n_fft=1024, sr=16000,

                 f_min=0.0, f_max=None, device=None):

        self.mel_fb = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,

                                fmin=f_min, fmax=f_max).astype(np.float32)



    def __call__(self, spec_f):

        spec_m = to_device(torch.from_numpy(self.mel_fb), spec_f.device) @ spec_f

        assert(spec_m.dim() == 4) # Check dim (n sample * channels * h * w)

        return spec_m





class ToDecibels(MySoundToImage):

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

            max_spec.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)

            spec_db = torch.max(spec_db, max_spec - self.top_db)

            if self.normalized:

                # normalize to [0, 1]

                spec_db.add_(self.top_db).div_(self.top_db)

        assert(spec_db.dim() == 4) # Check dim (n sample * channels * h * w)

        return spec_db
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
DATA = Path('../input')

CSV_TRN_CURATED = DATA/'train_curated.csv'

CSV_TRN_NOISY = DATA/'train_noisy.csv'

CSV_SUBMISSION = DATA/'sample_submission.csv'

TRN_CURATED = DATA/'train_curated'

TRN_NOISY = DATA/'train_noisy'

TEST = DATA/'test'



WORK = Path('work')

IMG_TRN_CURATED = WORK/'image/trn_curated'

IMG_TRN_NOISY = WORK/'image/trn_curated'

IMG_TEST = WORK/'image/test'

for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST]: 

    Path(folder).mkdir(exist_ok=True, parents=True)



df = pd.read_csv(CSV_TRN_CURATED)

df_n = pd.read_csv(CSV_TRN_NOISY)

test_df = pd.read_csv(CSV_SUBMISSION)
n_fft = 512 # output of fft will have shape [513 x n_frames]

n_hop = 94  # width of Spectogram = max_seconds * sample rate / n_hop

n_mels = 128 # Height of spectogram

sample_rate = 48127

max_seconds = 2

f_min=0

f_max=8000

noise_scl=0.005





train_tfms = get_data_augmentation_transforms(sample_rate=sample_rate, max_seconds=max_seconds, 

                                              noise_scl=noise_scl)

valid_tfms = get_data_augmentation_transforms(sample_rate=sample_rate, max_seconds=max_seconds)



dl_tfms = get_frequency_batch_transforms(n_fft=n_fft, n_hop=n_hop,

                                            n_mels=n_mels, 

                                            f_min=f_min, f_max=f_max,

                                            sample_rate=sample_rate)
batch_size = 32



audios = (AudioItemList.from_df(df=df, path=DATA, folder='/train_curated', using_librosa=True)

          .split_by_rand_pct(0.1)

          .label_from_df(label_delim=',')

          .add_test_folder('test')

          .transform(tfms=(train_tfms, valid_tfms))

          .databunch(bs=batch_size, tfms=dl_tfms)

         ).normalize()
audios.c, len(audios.train_ds), len(audios.valid_ds), len(audios.test_ds)

xs, ys = audios.one_batch()

print(xs.shape, ys.shape)

del xs, ys
audios.show_batch(3, ds_type=DatasetType.Train, figsize=(10, 5))
f_score = partial(fbeta, thresh=0.2)

acc_02 = partial(accuracy_thresh, thresh=0.2)
learn = create_cnn(audios, models.xresnet50, pretrained=False, metrics=[f_score, acc_02], model_dir='../../')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, 3e-2)
preds, _ = learn.TTA(ds_type=DatasetType.Test)
for i, v in enumerate(learn.data.classes):

    test_df[v] = preds[:, i]



test_df.to_csv('submission.csv', index=False)

test_df.head()
print(f"Kernel run time = {(time.time()-start_time)/3600} hours")