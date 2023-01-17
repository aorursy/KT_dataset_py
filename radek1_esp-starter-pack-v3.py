!pip install /kaggle/input/pyfftwwheel/pyFFTW-0.12.0-cp37-cp37m-manylinux1_x86_64.whl
import os

from pathlib import Path

import pandas as pd

import librosa

import matplotlib.pyplot as plt

import soundfile as sf

from torch.utils.data import Dataset, DataLoader

import numpy as np

import torch

import torchvision

from torch import nn

import warnings

warnings.filterwarnings("ignore", category=UserWarning)



from pyfftw.builders import rfft as rfft_builder

from pyfftw import empty_aligned



MEL_BANDS=80

MEL_MIN=27.5

MEL_MAX=10000

SAMPLE_RATE=32_000

THRESHOLD = 0.98
classes = pd.read_pickle('../input/espstarterpackv3/classes.pkl')
def lme_pool(x, alpha=1.0): # log-mean-exp pool

    '''alpha -> approximates maxpool, alpha -> 0 approximates mean pool'''

    T = x.shape[1]

    mult_log = torch.log(torch.tensor(1/T))

    return 1/alpha * (mult_log + torch.logsumexp((alpha * x), dim=1))
# class Model(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.cnn = nn.Sequential(*list(torchvision.models.resnet34(False).children())[:-2])

#         self.classifier = nn.Sequential(*[

#             nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.BatchNorm1d(512),

#             nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.BatchNorm1d(512),

#             nn.Linear(512, len(classes))

#         ])

    

#     def forward(self, x):

#         bs, im_num, ch, y_dim, x_dim = x.shape

#         x = self.cnn(x.view(-1, ch, y_dim, x_dim))

#         x = x.mean((2,3))

#         x = self.classifier(x)

#         x = x.view(bs, im_num, -1)

#         x = lme_pool(x)

#         return x
class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(*list(torchvision.models.resnet34(False).children())[:-2])

        self.classifier = nn.Sequential(*[

            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.BatchNorm1d(512),

            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.BatchNorm1d(512),

            nn.Linear(512, len(classes))

        ])

        self.register_parameter('alpha', nn.Parameter(torch.tensor(1.)))

    

    def forward(self, x):

        bs, im_num, ch, y_dim, x_dim = x.shape

        x = self.cnn(x.view(-1, ch, y_dim, x_dim))

        x = x.mean((2,3))

        x = self.classifier(x)

        x = x.view(bs, im_num, -1)

        x = lme_pool(x, self.alpha)

        return x
model = Model()

model.load_state_dict(torch.load('../input/espstarterpackv3/129_lmepool_learned_alpha_0.56.pth'))

model.cuda()

model.eval();
def spectrogram(samples, sample_rate, frame_len, fps, batch=48, dtype=None,

                bins=None, plans=None):

    """

    Computes a magnitude spectrogram for a given vector of samples at a given

    sample rate (in Hz), frame length (in samples) and frame rate (in Hz).

    Allows to transform multiple frames at once for improved performance (with

    a default value of 48, more is not always better). Returns a numpy array.

    Allows to return a limited number of bins only, with improved performance

    over discarding them afterwards. Optionally accepts a set of precomputed

    plans created with spectrogram_plans(), required when multi-threading.

    """

    if dtype is None:

        dtype = samples.dtype

    if bins is None:

        bins = frame_len // 2 + 1

    if len(samples) < frame_len:

        return np.empty((0, bins), dtype=dtype)

    if plans is None:

        plans = spectrogram_plans(frame_len, batch, dtype)

    rfft1, rfft, win = plans

    hopsize = int(sample_rate // fps)

    num_frames = (len(samples) - frame_len) // hopsize + 1

    nabs = np.abs

    naa = np.asanyarray

    if batch > 1 and num_frames >= batch and samples.flags.c_contiguous:

        frames = np.lib.stride_tricks.as_strided(

                samples, shape=(num_frames, frame_len),

                strides=(samples.strides[0] * hopsize, samples.strides[0]))

        spect = [nabs(rfft(naa(frames[pos:pos + batch:], dtype) * win)[:, :bins])

                 for pos in range(0, num_frames - batch + 1, batch)]

        samples = samples[(num_frames // batch * batch) * hopsize::]

        num_frames = num_frames % batch

    else:

        spect = []

    if num_frames:

        spect.append(np.vstack(

                [nabs(rfft1(naa(samples[pos:pos + frame_len:],

                                dtype) * win)[:bins:])

                 for pos in range(0, len(samples) - frame_len + 1, hopsize)]))

    return np.vstack(spect) if len(spect) > 1 else spect[0]





def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq,

                          max_freq):

    """

    Creates a mel filterbank of `num_bands` triangular filters, with the first

    filter starting at `min_freq` and the last one stopping at `max_freq`.

    Returns the filterbank as a matrix suitable for a dot product against

    magnitude spectra created from samples at a sample rate of `sample_rate`

    with a window length of `frame_len` samples.

    """

    # prepare output matrix

    input_bins = (frame_len // 2) + 1

    filterbank = np.zeros((input_bins, num_bands))



    # mel-spaced peak frequencies

    min_mel = 1127 * np.log1p(min_freq / 700.0)

    max_mel = 1127 * np.log1p(max_freq / 700.0)

    spacing = (max_mel - min_mel) / (num_bands + 1)

    peaks_mel = min_mel + np.arange(num_bands + 2) * spacing

    peaks_hz = 700 * (np.exp(peaks_mel / 1127) - 1)

    fft_freqs = np.linspace(0, sample_rate / 2., input_bins)

    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)



    # fill output matrix with triangular filters

    for b, filt in enumerate(filterbank.T):

        # The triangle starts at the previous filter's peak (peaks_freq[b]),

        # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].

        left_hz, top_hz, right_hz = peaks_hz[b:b + 3]  # b, b+1, b+2

        left_bin, top_bin, right_bin = peaks_bin[b:b + 3]

        # Create triangular filter compatible to yaafe

        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) /

                                  (top_bin - left_bin))

        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) /

                                   (right_bin - top_bin))

        filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()



    return filterbank



def spectrogram_plans(frame_len, batch=48, dtype=np.float32):

    """

    Precompute plans for spectrogram(), for a given frame length, batch size

    and dtype. Returns two plans (single spectrum and batch), and a window.

    """

    input_array = empty_aligned((batch, frame_len), dtype=dtype)

    win = np.hanning(frame_len).astype(dtype)

    return (rfft_builder(input_array[0]), rfft_builder(input_array), win)
filterbank = create_mel_filterbank(SAMPLE_RATE, 256, MEL_BANDS, MEL_MIN, MEL_MAX)



def audio_to_melspec(audio):

    spec = spectrogram(audio, SAMPLE_RATE, 256, 128)

    return (spec @ filterbank).T
TEST_PATH = Path('../input/birdsong-recognition') if os.path.exists('../input/birdsong-recognition/test_audio') else Path('../input/birdcall-check')



TEST_AUDIO_PATH = TEST_PATH/'test_audio'

test_df = pd.read_csv(TEST_PATH/'test.csv')

test_df.head()
class AudioDataset(Dataset):

    def __init__(self, items, classes, rec, example_dur_mult=1):

        self.items = items

        self.vocab = classes

        self.rec = rec

        self.example_dur_mult = example_dur_mult

    

    def __getitem__(self, idx):

        _, rec_fn, start = self.items[idx]

        x = self.rec[start*SAMPLE_RATE:(start+(5 * self.example_dur_mult))*SAMPLE_RATE]

        example = self.get_specs(x)

        example = self.normalize(example)

        imgs = example.reshape(-1, 3, 80, 212)

        return imgs.astype(np.float32)

    

    def get_specs(self, x):

        xs = []

        for i in range(self.example_dur_mult * 3):

            start_frame = int(i * 1.66 * SAMPLE_RATE)

            xs.append(x[start_frame:start_frame+int(1.66*SAMPLE_RATE)])



        specs = []

        for x in xs:

            specs.append(audio_to_melspec(x))

        return np.stack(specs)

    

    def normalize(self, example):

        return (example - 0.12934518) / 0.5612393

    

    def show(self, idx):

        x = self[idx][0]

        return plt.imshow(x.transpose(1,2,0)[:, :, 0])

    

    def __len__(self):

        return len(self.items)
%%time



row_ids = []

results = []



for audio_id in test_df[test_df.site.isin(['site_1', 'site_2'])].audio_id.unique():

    items = [(row.row_id, row.audio_id, int(row.seconds)-5) for idx, row in test_df[test_df.audio_id == audio_id].iterrows()]

    rec = librosa.load(TEST_AUDIO_PATH/f'{audio_id}.mp3', sr=SAMPLE_RATE, res_type='kaiser_fast')[0]

    test_ds = AudioDataset(items, classes, rec)

    dl = DataLoader(test_ds, batch_size=64)

    for batch in dl:

        with torch.no_grad():

            preds = model(batch.cuda()).cpu().detach()

            for row in preds:

                birds = []

                for idx in np.where(row > THRESHOLD)[0]:

                    birds.append(classes[idx])

                if not birds: birds = ['nocall']

                results.append(' '.join(birds)) 

    row_ids += [item[0] for item in items]
%%time



EXAMPLE_DUR_MULT = 10 # example duration = 5 sec x EXAMPLE_DUR_MULT

for audio_id in test_df[test_df.site=='site_3'].audio_id.unique():

    rec = librosa.load(TEST_AUDIO_PATH/f'{audio_id}.mp3', sr=SAMPLE_RATE, res_type='kaiser_fast')[0]

    current_row = test_df[test_df.audio_id == audio_id]

    duration = rec.shape[0] // SAMPLE_RATE

    mod_duration = duration - (duration % (5 * EXAMPLE_DUR_MULT))

    items = [(current_row.row_id.item(), current_row.audio_id.item(), start_sec) for start_sec in [0 + i * 5 * EXAMPLE_DUR_MULT for i in range(mod_duration // (5 * EXAMPLE_DUR_MULT))]]

    if duration > EXAMPLE_DUR_MULT * 5: items.append((current_row.row_id.item(), current_row.audio_id.item(), duration-(EXAMPLE_DUR_MULT * 5)))

    test_ds = AudioDataset(items, classes, rec, EXAMPLE_DUR_MULT)

    dl = DataLoader(test_ds, batch_size=64)

    

    birds = []

    for batch in dl:

        with torch.no_grad():

            preds = model(batch.cuda()).cpu().detach()

            for row in preds:

                for idx in np.where(row > THRESHOLD)[0]:

                    birds.append(classes[idx])

    row_ids.append(current_row.row_id.item())

    if not birds: birds = ['nocall']

    results.append(' '.join(list(set(birds))))             
predicted = pd.DataFrame(data={'row_id': row_ids, 'birds': results})

sub = pd.DataFrame(data={'row_id': test_df.row_id})

sub = sub.merge(predicted, 'left', 'row_id')

sub.to_csv('submission.csv', index=False)