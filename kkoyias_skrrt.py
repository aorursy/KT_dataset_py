import os

from shutil import copyfile as cp



SCRIPTS = ['data.py', 'config.py', 'models.py']

for f in SCRIPTS:

    cp(os.path.join('../input/myinput', f), f)

print('Scripts loaded!')
print(' > Installing requirements...')

!pip install --upgrade pip

!pip install torch

!pip install textgrid

!apt-get install -y libsndfile-dev

!pip install soundfile



print('\033[1;32mDone!\033[0m')
print(' > Importing...', end='')

import os

import data

import torch

import models

import pandas as pd

import soundfile as sf

from config import *

from tqdm import tqdm

print('\033[1;32mdone!\033[0m')
def predict(wav):

    signal, _ = sf.read(wav)

    signal = torch.tensor(signal, device=device).float().unsqueeze(0)

    label = model.decode_intents(signal)

    return label



def set_label(category, intents):

    category = intents.loc[intents.intent == category]

    return UNSURE if category.empty else category.category.item()

print('Well defined!')
UNSURE = 31

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = data.read_config('../input/myinput/no_unfreezing/no_unfreezing.cfg'); _,_,_=data.get_SLU_datasets(config)

model = models.Model(config).eval()

model.load_state_dict(torch.load('../input/myinput/no_unfreezing/model_state.pth', map_location=device)) # load trained model
TEST = '../input/myinput/test.csv'

SPEAKERS = '../input/myinput/speakers'

test = pd.read_csv(TEST)

df, paths = list(), list()

files = set(test['file'].apply(lambda f: f.replace('.png', '.wav')))

for i, speaker in enumerate(os.listdir(SPEAKERS)):

    speaker = os.path.join(SPEAKERS, speaker)

    for wav in os.listdir(speaker):

        if wav not in files:

            continue

        wav = os.path.join(speaker, wav)

        paths.append(wav)



df = pd.DataFrame({'file': paths})

tqdm.pandas(desc='Predicting command labels')

df['category'] = df['file'].progress_apply(lambda f: predict(f))



df = pd.DataFrame(df, columns=['file', 'category'])

df['category'] = df['category'].apply(lambda l: ','.join(l[0]))
INTENTS = '../input/myinput/intents.csv'

intents = pd.read_csv(INTENTS)

tqdm.pandas(desc='Mapping intent to category ID', total=df.shape[0])

df['category'] = df['category'].progress_apply(lambda c: set_label(c, intents))
df['file'] = df['file'].apply(lambda file: os.path.basename(file).replace('.wav', '.png'))

tqdm.pandas(desc='Mapping files to category ID', total=test.shape[0])

test['category'] = test['file'].progress_apply(lambda file: df.loc[df.file == file]['category'].item())
test['file'] = range(1, test['file'].shape[0] + 1)

test = test.rename(columns={'file': 'id'})



SUBMISSION = 'submission.csv'

test.to_csv(SUBMISSION, index=False)

print('Submission ready!!!')
# load our predictions as well as the ground truth and sort by id

LABELS = '../input/myinput/1.csv'

sub = pd.read_csv(SUBMISSION).sort_values(by='id')['category']

labels = pd.read_csv(LABELS).sort_values(by='id')['category']



# compare

correct = (sub == labels).sum()

total = labels.shape[0]

print(f'\033[1;32mAccuracy\033[0m: {correct/total:.6f}')
for f in SCRIPTS + [SUBMISSION]:

    os.remove(f)

print('All clean!')