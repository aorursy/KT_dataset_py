import librosa

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from matplotlib import pyplot as plt

import math



%matplotlib inline
slice_len   = 2

slice_count = 1

sr          = 44100

n_mels      = 256

fmin        = 20

hop_length  = int(sr/(n_mels/slice_len)) # ensures square mel-spectrogram slice

fmax        = sr//2



y = librosa.effects.trim(librosa.load('../input/train_noisy/42f7abb4.wav' , sr)[0])[0]



s = librosa.feature.melspectrogram(y, 

                                   sr         = sr,

                                   n_mels     = n_mels,

                                   hop_length = hop_length,

                                   n_fft      = n_mels*20,

                                   fmin       = fmin,

                                   fmax       = fmax)
gain          = 0.6

bias          = 0.1 

power         = 0.2 

time_constant = 0.4 

eps           = 1e-9



time_constant = 0.4



power_to_db = librosa.power_to_db(s)



pcen_librosa = librosa.core.pcen(s, 

                                 sr            = sr,

                                 hop_length    = hop_length,

                                 gain          = gain,

                                 bias          = bias,

                                 power         = power,

                                 time_constant = time_constant,

                                 eps           = eps)



fig = plt.figure(figsize=(20,3))

fig.suptitle("Power to Db")

plt.imshow(power_to_db)





fig = plt.figure(figsize=(20,3))

fig.suptitle("PCEN")

plt.imshow(pcen_librosa)
def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):

    frames = x.split(1, -2)

    m_frames = []

    last_state = None

    for frame in frames:

        if last_state is None:

            last_state = s * frame

            m_frames.append(last_state)

            continue

        if training:

            m_frame = ((1 - s) * last_state).add_(s * frame)

        else:

            m_frame = (1 - s) * last_state + s * frame

        last_state = m_frame

        m_frames.append(m_frame)

    M = torch.cat(m_frames, 1)

    if training:

        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r

    else:

        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)

    return pcen_





class PCENTransform(nn.Module):



    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):

        super().__init__()

        if trainable:

            self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))

            self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))

            self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))

            self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))

        else:

            self.s = s

            self.alpha = alpha

            self.delta = delta

            self.r = r

        self.eps = eps

        self.trainable = trainable



    def forward(self, x):

        x = x.permute((0,1,3,2)).squeeze(dim=1)

        if self.trainable:

            x = pcen(x, self.eps, torch.exp(self.log_s), torch.exp(self.log_alpha), torch.exp(self.log_delta), torch.exp(self.log_r), self.training and self.trainable)

        else:

            x = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable)

        x = x.unsqueeze(dim=1).permute((0,1,3,2))

        return x
t = torch.tensor(s)



T = time_constant * sr / hop_length



b = (math.sqrt(1 + 4* T**2) - 1) / (2 * T**2)    # as per librosa documentation



pcen_torch = pcen(t[None,...].permute((0,2,1)),  # change the shape of the mels appropriately for the PyTorch pcen function

                  eps      = eps, 

                  s        = b, 

                  alpha    = gain,

                  delta    = bias, 

                  r        = power, 

                  training = True)



pcen_torch = pcen_torch.permute((0,2,1)).squeeze().numpy()     # change the shape back and convert to numpy array from tensor
np.allclose(pcen_librosa, pcen_torch)
width  = 256

start  = np.random.randint(0,s.shape[1]-width) 

end    = start + width



crop_then_slice = librosa.core.pcen(s[:,start:end],

                                    sr            = sr,

                                    hop_length    = hop_length,

                                    gain          = gain,

                                    bias          = bias,

                                    power         = power,

                                    time_constant = time_constant,

                                    eps           = eps)



slice_then_crop = pcen_librosa[:,start:end]
np.allclose(crop_then_slice, slice_then_crop)
fig, ax = plt.subplots(1,3,figsize=[20,9])

ax[0].set_title("Crop then slice")

ax[0].imshow(crop_then_slice)



ax[1].set_title("Slice then crop")

ax[1].imshow(slice_then_crop)



ax[2].set_title("Difference")

ax[2].imshow(crop_then_slice-slice_then_crop)