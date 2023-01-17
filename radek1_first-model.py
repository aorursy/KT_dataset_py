!pip install /kaggle/input/fastai2-wheels/fastscript-0.1.4-py3-none-any.whl

!pip install /kaggle/input/fastai2-wheels/nbdev-0.2.12-py3-none-any.whl

!pip install /kaggle/input/fastai2-wheels/fastprogress-0.2.2-py3-none-any.whl

!pip install /kaggle/input/fastai2-wheels/fastcore-0.1.10-py3-none-any.whl

!pip install /kaggle/input/fastai2-wheels/fastai2-0.0.10-py3-none-any.whl
import pandas as pd

import torch

from fastai2.vision.all import *

import soundfile as sf

from pathlib import Path

import librosa

import multiprocessing

import warnings

warnings.filterwarnings("ignore", category=UserWarning)



from torch.utils.data import Dataset

from torch.utils.data import DataLoader
mean, std = (-6.132126808166504e-05, 0.04304003225515465)

classes = pd.read_pickle('../input/birdcall-first-model/classes.pkl')



get_arch = lambda: nn.Sequential(*[

    Lambda(lambda x: x.unsqueeze(1)),

    ConvLayer(1, 16, ks=64, stride=2, ndim=1),

    ConvLayer(16, 16, ks=8, stride=8, ndim=1),

    ConvLayer(16, 32, ks=32, stride=2, ndim=1),

    ConvLayer(32, 32, ks=8, stride=8, ndim=1),

    ConvLayer(32, 64, ks=16, stride=2, ndim=1),

    ConvLayer(64, 128, ks=8, stride=2, ndim=1),

    ConvLayer(128, 256, ks=4, stride=2, ndim=1),

    ConvLayer(256, 256, ks=4, stride=4, ndim=1),

    Flatten(),

    LinBnDrop(5120, 512, p=0.25, act=nn.ReLU()),

    LinBnDrop(512, 512, p=0.25, act=nn.ReLU()),

    LinBnDrop(512, 256, p=0.25, act=nn.ReLU()),

    LinBnDrop(256, len(classes)),

    nn.Sigmoid()

])
model = get_arch()

model.load_state_dict(torch.load('../input/birdcall-first-model/first_model.pth'))

model.cuda()

model.eval();
SAMPLE_RATE = 32_000



TEST_PATH = Path('../input/birdsong-recognition') if os.path.exists('../input/birdsong-recognition/test_audio') else Path('../input/birdcall-check')



TEST_AUDIO_PATH = TEST_PATH/'test_audio'

test_df = pd.read_csv(TEST_PATH/'test.csv')

test_df.head()
class AudioDataset(Dataset):

    def __init__(self, items, classes, rec, mean=None, std=None):

        self.items = items

        self.vocab = classes

        self.do_norm = (mean and std)

        self.mean = mean

        self.std = std

        self.rec = rec

    def __getitem__(self, idx):

        _, rec_fn, start = self.items[idx]

        x = self.rec[start*SAMPLE_RATE:(start+5)*SAMPLE_RATE]

        if self.do_norm: x = self.normalize(x)

        return x.astype(np.float32)

    def normalize(self, x):

        return (x - self.mean) / self.std    

    def __len__(self):

        return len(self.items)
%%time



row_ids = []

results = []



for audio_id in test_df[test_df.site.isin(['site_1', 'site_2'])].audio_id.unique():

    items = [(row.row_id, row.audio_id, int(row.seconds)-5) for idx, row in test_df[test_df.audio_id == audio_id].iterrows()]

    rec = librosa.load(TEST_AUDIO_PATH/f'{audio_id}.mp3', sr=SAMPLE_RATE, res_type='kaiser_fast')[0]

    test_ds = AudioDataset(items, classes, rec, mean=mean, std=std)

    dl = DataLoader(test_ds, batch_size=128)

    for batch in dl:

        with torch.no_grad():

            preds = model(batch.cuda()).cpu().detach()

            for row in preds:

                birds = []

                for idx in np.where(row > 0.5)[0]:

                    birds.append(classes[idx])

                if not birds: birds = ['nocall']

                results.append(' '.join(birds)) 

    row_ids += [item[0] for item in items]
predicted = pd.DataFrame(data={'row_id': row_ids, 'birds': results})



sub = pd.DataFrame(data={'row_id': test_df.row_id})

sub = sub.merge(predicted, 'left', 'row_id')

sub.fillna('nocall', inplace=True)

sub.to_csv('submission.csv', index=False)