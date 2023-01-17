import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
TRAIN_DIR = Path('../input/birdcalldatasetnpy/train_resampled_npy')

SAMPLE_RATE = 32_000
plt.imshow(np.load('../input/birdcalldatasetnpy/train_resampled_npy/aldfly/XC135454.npy')[:, :256]);
!pip install /kaggle/input/pyfftwwheel/pyFFTW-0.12.0-cp37-cp37m-manylinux1_x86_64.whl



from pyfftw.builders import rfft as rfft_builder

from pyfftw import empty_aligned



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



filterbank = create_mel_filterbank(SAMPLE_RATE, 256, 80, 27.5, 10000)



def audio_to_melspec(audio):

    spec = spectrogram(audio, SAMPLE_RATE, 256, 128)

    return (spec @ filterbank).T
BS = 100

MAX_LR = 1e-3



classes = [directory.name for directory in TRAIN_DIR.iterdir()]

train_items = []



for directory in TRAIN_DIR.iterdir():

    ebird_code = directory.name

    for recording in directory.iterdir():

        train_items.append((ebird_code, recording))
import torch

import torchvision



class TrainDataset(torch.utils.data.Dataset):    

    def __getitem__(self, idx):

        cls, path = train_items[idx]

        example = self.get_spec(path)

        return example, self.one_hot_encode(cls)

    

    def get_spec(self, path):

        frames_per_spec = 212  

        x = np.load(path)



        specs = []

        for _ in range(3):

            if x.shape[1] < frames_per_spec:

                spec = np.zeros((80, frames_per_spec))

                start_frame = np.random.randint(frames_per_spec-x.shape[1])

                spec[:, start_frame:start_frame+x.shape[1]] = x

            else:

                start_frame = int(np.random.rand() * (x.shape[1] - frames_per_spec))

                spec = x[:, start_frame:start_frame+frames_per_spec]

            specs.append(spec)

        

        return np.stack(specs).reshape(3, 80, frames_per_spec).astype(np.float32)

    

    def show(self, idx):

        x = self[idx][0]

        return plt.imshow(x.transpose(1,2,0)[:, :, 0])

        

    def one_hot_encode(self, cls):

        y = classes.index(cls)

        one_hot = np.zeros((len(classes)))

        one_hot[y] = 1

        return one_hot

    def __len__(self):

        return len(train_items)

    

train_ds = TrainDataset()



import multiprocessing

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BS, num_workers=multiprocessing.cpu_count(), pin_memory=True, shuffle=True)
import torchvision



pretrained_res34 = torchvision.models.resnet34(False)

pretrained_res34.load_state_dict(torch.load('../input/pretrained-pytorch/resnet34-333f7ec4.pth'))



class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.bn = torch.nn.BatchNorm2d(3)

        self.cnn = torch.nn.Sequential(*list(pretrained_res34.children())[:-2], torch.nn.AdaptiveMaxPool2d(1))

        self.classifier = torch.nn.Sequential(*[

            torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Dropout(p=0.2), torch.nn.BatchNorm1d(512),

            torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Dropout(p=0.2), torch.nn.BatchNorm1d(512),

            torch.nn.Linear(512, len(classes))

        ])

    

    def forward(self, x):

        max_per_example = x.view(x.shape[0], -1).max(1)[0]

        x[max_per_example != 0] /= max_per_example[max_per_example != 0][:, None, None, None]

        x = self.cnn(x)[:, :, 0, 0]

        x = self.classifier(x)

        return x



model = Model().cuda()
for param in model.cnn.parameters(): param.requires_grad = False
def train(num_epochs):

    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(num_epochs):

        for data in train_dl:

            inputs, labels = data[0].cuda(), data[1].cuda()

            optimizer.zero_grad()



            outputs = model(inputs)

            loss = criterion(outputs, labels)



            if np.isnan(loss.item()): raise Exception(f'!!! nan encountered in loss !!! epoch: {epoch}\n')

            loss.backward()

            optimizer.step()

            scheduler.step()
%%time



train(30)
%%time



for param in model.cnn.parameters(): param.requires_grad = True



train(60)
import os

import pandas as pd



TEST_PATH = Path('../input/birdsong-recognition') if os.path.exists('../input/birdsong-recognition/test_audio') else Path('../input/birdcall-check')



TEST_AUDIO_PATH = TEST_PATH/'test_audio'

test_df = pd.read_csv(TEST_PATH/'test.csv')
import librosa



class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, items, classes, rec):

        self.items = items

        self.vocab = classes

        self.rec = rec

    

    def __getitem__(self, idx):

        _, rec_fn, start = self.items[idx]

        x = self.rec[start*SAMPLE_RATE:(start+5)*SAMPLE_RATE]

        example = self.get_specs(x)

        return example.astype(np.float32)

    

    def get_specs(self, x):

        xs = []

        for i in range(3):

            start_frame = int(i * 1.66 * SAMPLE_RATE)

            xs.append(x[start_frame:start_frame+int(1.66*SAMPLE_RATE)])



        specs = []

        for x in xs:

            specs.append(audio_to_melspec(x))

        return np.stack(specs).reshape(3, 80, 212)

    

    def show(self, idx):

        x = self[idx][0]

        return plt.imshow(x.transpose(1,2,0)[:, :, 0])

    

    def __len__(self):

        return len(self.items)
%%time



import warnings

warnings.filterwarnings("ignore", category=UserWarning)



row_ids = []

all_preds = []



model.eval()

for audio_id in test_df[test_df.site.isin(['site_1', 'site_2'])].audio_id.unique():

    items = [(row.row_id, row.audio_id, int(row.seconds)-5) for idx, row in test_df[test_df.audio_id == audio_id].iterrows()]

    rec = librosa.load(TEST_AUDIO_PATH/f'{audio_id}.mp3', sr=SAMPLE_RATE, res_type='kaiser_fast')[0]

    test_ds = AudioDataset(items, classes, rec)

    dl = torch.utils.data.DataLoader(test_ds, batch_size=64)

    for batch in dl:

        with torch.no_grad():

            preds = model(batch.cuda()).sigmoid().cpu().detach()

            all_preds.append(preds)

    row_ids += [item[0] for item in items]
%%time



for audio_id in test_df[test_df.site=='site_3'].audio_id.unique():

    rec = librosa.load(TEST_AUDIO_PATH/f'{audio_id}.mp3', sr=SAMPLE_RATE, res_type='kaiser_fast')[0]

    current_row = test_df[test_df.audio_id == audio_id].iloc[0] # assuming only one row per recording for site_3

    duration = rec.shape[0] // SAMPLE_RATE

    items = [(current_row.row_id, current_row.audio_id, start_sec) for start_sec in [0 + i * 5 for i in range(duration // 5)]]

    test_ds = AudioDataset(items, classes, rec)

    dl = torch.utils.data.DataLoader(test_ds, batch_size=64)

    

    preds_for_site = []

    for batch in dl:

        with torch.no_grad():

            preds = model(batch.cuda()).sigmoid().cpu().detach()

            preds_for_site.append(preds)

    

    if preds_for_site:

        row_ids.append(current_row.row_id)

        all_preds.append(torch.cat(preds_for_site).max(0)[0].unsqueeze(0))     
all_preds = torch.cat(all_preds)
%%time

thresh = 1

minimum_prediction_rate = 0.04



while (all_preds > thresh).any(1).float().mean() < minimum_prediction_rate:

    thresh -= 0.001
results = []



for row in all_preds:

    birds = []

    for idx in np.where(row > thresh)[0]:

        birds.append(classes[idx])

    if not birds: birds = ['nocall']

    results.append(' '.join(birds))
predicted = pd.DataFrame(data={'row_id': row_ids, 'birds': results})

predicted.to_csv('submission.csv', index=False)