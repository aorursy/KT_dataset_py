import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import librosa
import time
import torchaudio
import torch

import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
import torch
import numpy as np
from scipy import special


class Resampler(torch.nn.Module):
    """
    Efficiently resample audio signals
    This module is much faster than resampling with librosa because it exploits pytorch's efficient conv1d operations
    This module is also faster than the existing pytorch resample function in
    https://github.com/pytorch/audio/blob/b6a61c3f7d0267c77f8626167cc1eda0335f2753/torchaudio/compliance/kaldi.py#L892
    
    Based on 
    https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py
    with improvements to include additional filter types and input parameters that align with the librosa api
    """

    def __init__(self,
                 input_sr, output_sr, dtype,
                 num_zeros = 64, cutoff_ratio = 0.95, filter='kaiser', beta=14.0):
        super().__init__()  # init the base class
        """
        This creates an object that can apply a symmetric FIR filter
        based on torch.nn.functional.conv1d.

        Args:
          input_sr:  The input sampling rate, AS AN INTEGER..
              does not have to be the real sampling rate but should
              have the correct ratio with output_sr.
          output_sr:  The output sampling rate, AS AN INTEGER.
              It is the ratio with the input sampling rate that is
              important here.
          dtype:  The torch dtype to use for computations (would be preferrable to 
               set things up so passing the dtype isn't necessary)
          num_zeros: The number of zeros per side in the (sinc*hanning-window)
              filter function.  More is more accurate, but 64 is already
              quite a lot. The kernel size is 2*num_zeros + 1.
          cutoff_ratio: The filter rolloff point as a fraction of the
             Nyquist frequency.
          filter: one of ['kaiser', 'kaiser_best', 'kaiser_fast', 'hann']
          beta: parameter for 'kaiser' filter

        You can think of this algorithm as dividing up the signals
        (input,output) into blocks where there are `input_sr` input
        samples and `output_sr` output samples.  Then we treat it
        using convolutional code, imagining there are `input_sr`
        input channels and `output_sr` output channels per time step.

        """
        assert isinstance(input_sr, int) and isinstance(output_sr, int)
        if input_sr == output_sr:
            self.resample_type = 'trivial'
            return
        
        def gcd(a, b):
            """ Return the greatest common divisor of a and b"""
            assert isinstance(a, int) and isinstance(b, int)
            if b == 0:
                return a
            else:
                return gcd(b, a % b)  

        d = gcd(input_sr, output_sr)
        input_sr, output_sr = input_sr // d, output_sr // d

        assert dtype in [torch.float32, torch.float64]
        assert num_zeros > 3  # a reasonable bare minimum
        np_dtype = np.float32 if dtype == torch.float32 else np.float64

        assert filter in ['hann', 'kaiser', 'kaiser_best', 'kaiser_fast']

        if filter == 'kaiser_best':
            num_zeros = 64
            beta = 14.769656459379492
            cutoff_ratio = 0.9475937167399596
            filter = 'kaiser'
        elif filter == 'kaiser_fast':
            num_zeros = 16
            beta = 8.555504641634386
            cutoff_ratio = 0.85
            filter = 'kaiser'

        # Define one 'block' of samples `input_sr` input samples
        # and `output_sr` output samples.  We can divide up
        # the samples into these blocks and have the blocks be
        #in correspondence.

        # The sinc function will have, on average, `zeros_per_block`
        # zeros per block.
        zeros_per_block = min(input_sr, output_sr) * cutoff_ratio

        # The convolutional kernel size will be n = (blocks_per_side*2 + 1),
        # i.e. we add that many blocks on each side of the central block.  The
        # window radius (defined as distance from center to edge)
        # is `blocks_per_side` blocks.  This ensures that each sample in the
        # central block can "see" all the samples in its window.
        #
        # Assuming the following division is not exact, adding 1
        # will have the same effect as rounding up.
        #blocks_per_side = 1 + int(num_zeros / zeros_per_block)
        blocks_per_side = int(np.ceil(num_zeros / zeros_per_block))

        kernel_width = 2*blocks_per_side + 1

        # We want the weights as used by torch's conv1d code; format is
        #  (out_channels, in_channels, kernel_width)
        # https://pytorch.org/docs/stable/nn.functional.html
        weights = torch.tensor((output_sr, input_sr, kernel_width), dtype=dtype)

        # Computations involving time will be in units of 1 block.  Actually this
        # is the same as the `canonical` time axis since each block has input_sr
        # input samples, so it would be one of whatever time unit we are using
        window_radius_in_blocks = blocks_per_side


        # The `times` below will end up being the args to the sinc function.
        # For the shapes of the things below, look at the args to `view`.  The terms
        # below will get expanded to shape (output_sr, input_sr, kernel_width) through
        # broadcasting
        # We want it so that, assuming input_sr == output_sr, along the diagonal of
        # the central block we have t == 0.
        # The signs of the output_sr and input_sr terms need to be opposite.  The
        # sign that the kernel_width term needs to be will depend on whether it's
        # convolution or correlation, and the logic is tricky.. I will just find
        # which sign works.


        times = (
            np.arange(output_sr, dtype=np_dtype).reshape((output_sr, 1, 1)) / output_sr -
            np.arange(input_sr, dtype=np_dtype).reshape((1, input_sr, 1)) / input_sr -
            (np.arange(kernel_width, dtype=np_dtype).reshape((1, 1, kernel_width)) - blocks_per_side))


        def hann_window(a):
            """
            hann_window returns the Hann window on [-1,1], which is zero
            if a < -1 or a > 1, and otherwise 0.5 + 0.5 cos(a*pi).
            This is applied elementwise to a, which should be a NumPy array.

            The heaviside function returns (a > 0 ? 1 : 0).
            """
            return np.heaviside(1 - np.abs(a), 0.0) * (0.5 + 0.5 * np.cos(a * np.pi))

        def kaiser_window(a, beta):
            w = special.i0(beta * np.sqrt(np.clip(1 - ((a - 0.0) / 1.0) ** 2.0, 0.0, 1.0))) / special.i0(beta)
            return np.heaviside(1 - np.abs(a), 0.0) * w


        # The weights below are a sinc function times a Hann-window function.
        #
        # Multiplication by zeros_per_block normalizes the sinc function 
        # (to compensate for scaling on the x-axis), so that the integral is 1.
        #
        # Division by input_sr normalizes the input function. Think of the input 
        # as a stream of dirac deltas passing through a low pass filter: 
        # in order to have the same magnitude as the original input function, 
        # we need to divide by the number of those deltas per unit time.
        if filter == 'hann':
            weights = (np.sinc(times * zeros_per_block)
                       * hann_window(times / window_radius_in_blocks)
                       * zeros_per_block / input_sr)
        else:
            weights = (np.sinc(times * zeros_per_block)
                       * kaiser_window(times / window_radius_in_blocks, beta)
                       * zeros_per_block / input_sr)

        self.input_sr = input_sr
        self.output_sr = output_sr

        # weights has dim (output_sr, input_sr, kernel_width).  
        # If output_sr == 1, we can fold the input_sr into the
        # kernel_width (i.e. have just 1 input channel); this will make the
        # convolution faster and avoid unnecessary reshaping.

        assert weights.shape == (output_sr, input_sr, kernel_width)
        if output_sr == 1:
            self.resample_type = 'integer_downsample'
            self.padding = input_sr * blocks_per_side
            weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
            self.weights = weights.transpose(1, 2).contiguous().view(1, 1, input_sr * kernel_width)

        elif input_sr == 1:
            # In this case we'll be doing conv_transpose, so we want the same weights that
            # we would have if we were *downsampling* by this factor-- i.e. as if input_sr,
            # output_sr had been swapped.
            self.resample_type = 'integer_upsample'
            self.padding = output_sr * blocks_per_side
            weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
            self.weights = weights.flip(2).transpose(0, 2).contiguous().view(1, 1, output_sr * kernel_width)
        else:
            self.resample_type = 'general'
            self.reshaped = False
            self.padding = blocks_per_side
            self.weights = torch.tensor(weights, dtype=dtype, requires_grad=False)

        self.weights = torch.nn.Parameter(self.weights, requires_grad=False)      

    @torch.no_grad()
    def forward(self, data):
        """
        Resample the data

        Args:
         input: a torch.Tensor with the same dtype as was passed to the
           constructor.
         There must be 2 axes, interpreted as (minibatch_size, sequence_length)...
         the minibatch_size may in practice be the number of channels.

        Return:  Returns a torch.Tensor with the same dtype as the input, and
         dimension (minibatch_size, (sequence_length//input_sr)*output_sr),
         where input_sr and output_sr are the corresponding constructor args,
         modified to remove any common factors.
        """
        if self.resample_type == 'trivial':
            return data
        elif self.resample_type == 'integer_downsample':
            (minibatch_size, seq_len) = data.shape
            # will be shape (minibatch_size, in_channels, seq_len) with in_channels == 1
            data = data.unsqueeze(1)
            data = torch.nn.functional.conv1d(data,
                                             self.weights,
                                             stride=self.input_sr,
                                             padding=self.padding)
            # shape will be (minibatch_size, out_channels = 1, seq_len);
            # return as (minibatch_size, seq_len)
            return data.squeeze(1)

        elif self.resample_type == 'integer_upsample':
            data = data.unsqueeze(1)
            data = torch.nn.functional.conv_transpose1d(data,
                                                      self.weights,
                                                      stride=self.output_sr,
                                                      padding=self.padding)

            return data.squeeze(1)
        else:
            assert self.resample_type == 'general'
            (minibatch_size, seq_len) = data.shape
            num_blocks = seq_len // self.input_sr
            if num_blocks == 0:
                # TODO: pad with zeros.
                raise RuntimeError("Signal is too short to resample")
            #data = data[:, 0:(num_blocks*self.input_sr)]  # Truncate input
            data = data[:, 0:(num_blocks*self.input_sr)].view(minibatch_size, num_blocks, self.input_sr)


            # Torch's conv1d expects input data with shape (minibatch, in_channels, time_steps), so transpose
            data = data.transpose(1, 2)


            data = torch.nn.functional.conv1d(data, self.weights,
                                          padding=self.padding)

            assert data.shape == (minibatch_size, self.output_sr, num_blocks)
            return data.transpose(1, 2).contiguous().view(minibatch_size, num_blocks * self.output_sr)



base_path = "../input/birdsong-recognition/"
train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
#convert the sample rates to int from string
train_df['sampling_rate'] = train_df['sampling_rate'].apply(lambda x: int(x.split(' ')[0]))
'''
sample rate census for the first 20 files
'''
train_df.loc[0:19, 'sampling_rate'].value_counts()
'''
load the first 20 audio files
'''

audio = {}
samplerates = {}
for i in range(20):
    fname = base_path + "train_audio/" + train_df["ebird_code"].iloc[i] +"/"+ train_df["filename"].iloc[i]
    y, sr = librosa.load(fname, sr=None)
    audio[i] = y
    samplerates[i] = sr
'''
resample with librosa, 'kaiser_best' settings
'''

t0 = time.time()
for i in range(20):
    y_librosa = librosa.resample(audio[i], samplerates[i], 22050, res_type='kaiser_best')
t1 = time.time()

print(f'execution time librosa resample (kaiser_best): {t1-t0}')    
'''
resample with efficient pytorch resampler
'kaiser_best' reproduces the 'kaiser_best' settings in librosa
'''

resampler = {}
resampler[44100] = Resampler(input_sr=44100, output_sr=22050, dtype=torch.float32, filter='kaiser_best')
resampler[48000] = Resampler(input_sr=48000, output_sr=22050, dtype=torch.float32, filter='kaiser_best')

t0 = time.time()
for i in range(20):
    if len(audio[i].shape) == 1:
        y = torch.tensor(audio[i]).unsqueeze(0)
    else:
        y = torch.tensor(audio[i])
   
    y_best = resampler[samplerates[i]].forward(y)
t1 = time.time()

print(f'execution time efficient torch resample (kaiser_best): {t1-t0}') 
'''
resample with torchaudio's resample
'''

resampler = {}
resampler[44100] = torchaudio.transforms.Resample(orig_freq=44100, new_freq=22050)
resampler[48000] = torchaudio.transforms.Resample(orig_freq=48000, new_freq=22050)

t0 = time.time()
for i in range(20):
    y_torchaudio = resampler[samplerates[i]].forward(torch.tensor(audio[i]))
t1 = time.time()

print(f'execution time torchaudio resample: {t1-t0}') 
'''
resample with efficient pytorch resampler
filter='hann' and num_zeros=6 reproduce the settings in torchaudio's existing resample function from above
'''

resampler = {}
resampler[44100] = Resampler(input_sr=44100, output_sr=22050, dtype=torch.float32, filter='hann', num_zeros=6)
resampler[48000] = Resampler(input_sr=48000, output_sr=22050, dtype=torch.float32, filter='hann', num_zeros=6)

t0 = time.time()
for i in range(20):
    if len(audio[i].shape) == 1:
        y = torch.tensor(audio[i]).unsqueeze(0)
    else:
        y = torch.tensor(audio[i])
  
    y_best_hann = resampler[samplerates[i]].forward(y)
t1 = time.time()

print(f'execution time efficient torch resample (hann, num_zeros=6): {t1-t0}')
print(y_librosa.shape)
print(y_best.size())
print(y_torchaudio.size())
print(y_best_hann.size())


torch.tensor(y_librosa)
print(np.max(y_librosa))
print(np.max(y_best.numpy()))
print(np.max(np.abs(y_librosa - y_best.squeeze(0).numpy())))
print(np.mean(np.abs(y_librosa - y_best.squeeze(0).numpy())))
plt.plot(np.arange(len(y_librosa)), y_librosa, 'k.')
plt.plot(np.arange(len(y_librosa)), y_best.squeeze(0).numpy(), 'r.')
plt.plot(y_best.squeeze(0).numpy(), y_best.squeeze(0).numpy()-y_librosa, 'k.')
plt.xlabel('y value efficient resample')
plt.ylabel('y value (efficient resample - librosa resample)')
print(np.max(y_torchaudio.squeeze(0).numpy()))
print(np.max(y_best_hann.numpy()))
print(np.max(np.abs(y_torchaudio.squeeze(0).numpy() - y_best_hann.squeeze(0).numpy())))
print(np.mean(np.abs(y_torchaudio.squeeze(0).numpy() - y_best_hann.squeeze(0).numpy())))
plt.plot(y_best_hann.squeeze(0).numpy(), y_best_hann.squeeze(0).numpy()-y_torchaudio.squeeze(0).numpy(), 'k.')
plt.xlabel('y value efficient resample')
plt.ylabel('y value (efficient resample - torchaudio resample)')
torch.sqrt(torch.mean(torch.pow(y_torchaudio, 2)))