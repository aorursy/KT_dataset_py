!pip install librosa
import numpy as np 

import pandas as pd 

import os

import warnings

warnings.filterwarnings("ignore")
import librosa

audio_data_ameavo = '../input/birdsong-recognition/train_audio/ameavo/XC133080.mp3' #take any sample input .mp3

audio_data_mouchi = '../input/birdsong-recognition/train_audio/mouchi/XC109939.mp3'

audio_data_nutwoo = '../input/birdsong-recognition/train_audio/nutwoo/XC109667.mp3'

audio_data_pygnut = '../input/birdsong-recognition/train_audio/pygnut/XC11739.mp3'

audio_data_vesspa = '../input/birdsong-recognition/train_audio/vesspa/XC102960.mp3'

audio_data_yetvir = '../input/birdsong-recognition/train_audio/yetvir/XC120867.mp3'

x_ameavo , sr = librosa.load(audio_data_ameavo)

x_mouchi , sr = librosa.load(audio_data_mouchi)

x_nutwoo , sr = librosa.load(audio_data_nutwoo)

x_pygnut , sr = librosa.load(audio_data_pygnut)

x_vesspa , sr = librosa.load(audio_data_vesspa)

x_yetvir , sr = librosa.load(audio_data_yetvir)



print(type(x_ameavo), type(sr))

print("Ameavo :", x_ameavo.shape, sr)

print("Mouchi :", x_mouchi.shape, sr)

print("Nutwoo :", x_nutwoo.shape, sr)

print("Pygnut :", x_pygnut.shape, sr)

print("Vesspa :", x_vesspa.shape, sr)

print("Yetvir :", x_yetvir.shape, sr)
librosa.load(audio_data_ameavo)

librosa.load(audio_data_mouchi)

librosa.load(audio_data_nutwoo)

librosa.load(audio_data_pygnut)

librosa.load(audio_data_vesspa)

librosa.load(audio_data_yetvir)
import IPython.display as ipd

print("Song of Ameavo :")

ipd.Audio(audio_data_ameavo)
print("Song of Mouchi :")

ipd.Audio(audio_data_mouchi)
print("Song of Nutwoo :")

ipd.Audio(audio_data_nutwoo)
print("Song of Pygnut :")

ipd.Audio(audio_data_pygnut)
print("Song of Vesspa :")

ipd.Audio(audio_data_vesspa)
print("Song of Yetvir :")

ipd.Audio(audio_data_yetvir)
# Changing default frequency

librosa.load(audio_data_vesspa, sr=85000)
ipd.Audio(audio_data_vesspa)
D_ameavo = librosa.stft(x_ameavo)

D_mouchi = librosa.stft(x_mouchi)

D_nutwoo = librosa.stft(x_nutwoo)

D_pygnut = librosa.stft(x_pygnut)

D_vesspa = librosa.stft(x_vesspa)

D_yetvir = librosa.stft(x_yetvir)

print("For Ameavo :", D_ameavo.shape, D_ameavo.dtype)

print("For Mouchi :", D_mouchi.shape, D_mouchi.dtype)

print("For Nutwoo :", D_nutwoo.shape, D_nutwoo.dtype)

print("For Pygnut :", D_pygnut.shape, D_pygnut.dtype)

print("For Vesspa :", D_vesspa.shape, D_vesspa.dtype)

print("For Yetvir :", D_yetvir.shape, D_yetvir.dtype)
S, phase = librosa.magphase(D_ameavo)

print(S.dtype, phase.dtype, np.allclose(D_ameavo, S * phase))
C = librosa.cqt(x_ameavo, sr=sr)

print(C.shape, C.dtype)
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import librosa.display



fig, ax = plt.subplots(6, figsize = (16, 12))

fig.suptitle('Sound Waves', fontsize=14)

librosa.display.waveplot(x_ameavo, sr = sr, ax=ax[0])

ax[0].set_ylabel('Ameavo')

librosa.display.waveplot(x_mouchi, sr = sr, ax=ax[1])

ax[1].set_ylabel('Mouchi')

librosa.display.waveplot(x_nutwoo, sr = sr, ax=ax[2])

ax[2].set_ylabel('Nutwoo')

librosa.display.waveplot(x_pygnut, sr = sr, ax=ax[3])

ax[3].set_ylabel('Pygnut')

librosa.display.waveplot(x_vesspa, sr = sr, ax=ax[4])

ax[4].set_ylabel('Vesspa')

librosa.display.waveplot(x_yetvir, sr = sr, ax=ax[5])

ax[5].set_ylabel('Yetvir')
X_ameavo = librosa.stft(x_ameavo)

X_mouchi = librosa.stft(x_mouchi)

X_nutwoo = librosa.stft(x_nutwoo)

X_pygnut = librosa.stft(x_pygnut)

X_vesspa = librosa.stft(x_vesspa)

X_yetvir = librosa.stft(x_yetvir)



Xdb_ameavo = librosa.amplitude_to_db(abs(X_ameavo))

Xdb_mouchi = librosa.amplitude_to_db(abs(X_mouchi))

Xdb_nutwoo = librosa.amplitude_to_db(abs(X_nutwoo))

Xdb_pygnut = librosa.amplitude_to_db(abs(X_pygnut))

Xdb_vesspa = librosa.amplitude_to_db(abs(X_vesspa))

Xdb_yetvir = librosa.amplitude_to_db(abs(X_yetvir))



fig, ax = plt.subplots(2, 3, figsize=(16, 10))

fig.suptitle('Spectrogram', fontsize=12)



librosa.display.specshow(Xdb_ameavo, sr = sr, x_axis = 'time', y_axis = 'hz', ax=ax[0, 0])

ax[0, 0].set_title('Ameavo', fontsize = 12)

librosa.display.specshow(Xdb_mouchi, sr = sr, x_axis = 'time', y_axis = 'hz', ax=ax[0, 1])

ax[0, 1].set_title('Mouchi', fontsize = 12)

librosa.display.specshow(Xdb_nutwoo, sr = sr, x_axis = 'time', y_axis = 'hz', ax=ax[0, 2])

ax[0, 2].set_title('Nutwoo', fontsize = 12)

librosa.display.specshow(Xdb_pygnut, sr = sr, x_axis = 'time', y_axis = 'hz', ax=ax[1, 0])

ax[1, 0].set_title('Pygnut', fontsize = 12)

librosa.display.specshow(Xdb_vesspa, sr = sr, x_axis = 'time', y_axis = 'hz', ax=ax[1, 1])

ax[1, 1].set_title('Vesspa', fontsize = 12)

librosa.display.specshow(Xdb_yetvir, sr = sr, x_axis = 'time', y_axis = 'hz', ax=ax[1, 2])

ax[1, 2].set_title('Yetvir', fontsize = 12)
fig, ax = plt.subplots(2, 3, figsize=(16, 10))

fig.suptitle('Spectrogram with Log Values of Frequency', fontsize=12)



librosa.display.specshow(Xdb_ameavo, sr=sr, x_axis='time', y_axis='log', ax=ax[0, 0])

ax[0, 0].set_title('Ameavo', fontsize = 12)

librosa.display.specshow(Xdb_mouchi, sr=sr, x_axis='time', y_axis='log', ax=ax[0, 1])

ax[0, 1].set_title('Mouchi', fontsize = 12)

librosa.display.specshow(Xdb_nutwoo, sr=sr, x_axis='time', y_axis='log', ax=ax[0, 2])

ax[0, 2].set_title('Nutwoo', fontsize = 12)

librosa.display.specshow(Xdb_pygnut, sr=sr, x_axis='time', y_axis='log', ax=ax[1, 0])

ax[1, 0].set_title('Pygnut', fontsize = 12)

librosa.display.specshow(Xdb_vesspa, sr=sr, x_axis='time', y_axis='log', ax=ax[1, 1])

ax[1, 1].set_title('Vesspa', fontsize = 12)

librosa.display.specshow(Xdb_yetvir, sr=sr, x_axis='time', y_axis='log', ax=ax[1, 2])

ax[1, 2].set_title('Yetvir', fontsize = 12)
sr = 22000 # sample rate

T = 5.0    # seconds

t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable

x = 0.5*np.sin(2*np.pi*220*t) # pure sine wave at 220 Hz

#Playing the audio

ipd.Audio(x, rate=sr) 

#Saving the audio

librosa.output.write_wav('tone_220.wav', x, sr)
import sklearn

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

spectral_centroids.shape

# Computing the time variable for visualization

plt.figure(figsize=(14, 6))

frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform

librosa.display.waveplot(x, sr=sr, alpha=0.3)

plt.plot(t, normalize(spectral_centroids), linewidth=1.5, color='r')
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

plt.figure(figsize=(12, 5))

librosa.display.waveplot(x, sr=sr, alpha=0.3)

plt.plot(t, normalize(spectral_rolloff), linewidth=1.5, color='r')
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]

spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]

spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]

spectral_bandwidth_5 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=5)[0]

plt.figure(figsize=(12, 6))

librosa.display.waveplot(x, sr=sr, alpha=0.3)

plt.plot(t, normalize(spectral_bandwidth_2), linewidth=1.5, color='r')

plt.plot(t, normalize(spectral_bandwidth_3), linewidth=1.5, color='g')

plt.plot(t, normalize(spectral_bandwidth_4), linewidth=1.5, color='y')

plt.plot(t, normalize(spectral_bandwidth_5), linewidth=1.5, color='b')

plt.legend(('p = 2', 'p = 3', 'p = 4', 'p = 5'))
#Plot the signal for Ameavo:

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x_ameavo, sr=sr)

plt.title('Normal Waveplot for Ameavo')

# Zooming in

n0 = 8800

n1 = 9000

plt.figure(figsize=(14, 6))

plt.plot(x_ameavo[n0:n1], linewidth=1.5)

plt.grid()

plt.title('Zoomed-In Waveplot for Ameavo')
#Plot the signal for Mouchi:

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x_mouchi, sr=sr)

plt.title('Normal Waveplot for Mouchi')

# Zooming in

n0 = 8800

n1 = 9000

plt.figure(figsize=(14, 6))

plt.plot(x_mouchi[n0:n1], linewidth=1.5)

plt.grid()

plt.title('Zoomed-In Waveplot for Mouchi')
#Plot the signal for Nutwoo:

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x_nutwoo, sr=sr)

plt.title('Normal Waveplot for Nutwoo')

# Zooming in

n0 = 8800

n1 = 9000

plt.figure(figsize=(14, 6))

plt.plot(x_nutwoo[n0:n1], linewidth=1.5)

plt.grid()

plt.title('Zoomed-In Waveplot for Nutwoo')
#Plot the signal for Pygnut:

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x_pygnut, sr=sr)

plt.title('Normal Waveplot for Pygnut')

# Zooming in

n0 = 8800

n1 = 9000

plt.figure(figsize=(14, 6))

plt.plot(x_pygnut[n0:n1], linewidth=1.5)

plt.grid()

plt.title('Zoomed-In Waveplot for Pygnut')
#Plot the signal for Vesspa:

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x_vesspa, sr=sr)

plt.title('Normal Waveplot for Vesspa')

# Zooming in

n0 = 8800

n1 = 9000

plt.figure(figsize=(14, 6))

plt.plot(x_vesspa[n0:n1], linewidth=1.5)

plt.grid()

plt.title('Zoomed-In Waveplot for Vesspa')
#Plot the signal for Yetvir:

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x_yetvir, sr=sr)

plt.title('Normal Waveplot for Yetvir')

# Zooming in

n0 = 8800

n1 = 9000

plt.figure(figsize=(14, 6))

plt.plot(x_yetvir[n0:n1], linewidth=1.5)

plt.grid()

plt.title('Zoomed-In Waveplot for Yetvir')
zero_crossings_ameavo = librosa.zero_crossings(x_ameavo[n0:n1], pad=False)

zero_crossings_mouchi = librosa.zero_crossings(x_mouchi[n0:n1], pad=False)

zero_crossings_nutwoo = librosa.zero_crossings(x_nutwoo[n0:n1], pad=False)

zero_crossings_pygnut = librosa.zero_crossings(x_pygnut[n0:n1], pad=False)

zero_crossings_vesspa = librosa.zero_crossings(x_vesspa[n0:n1], pad=False)

zero_crossings_yetvir = librosa.zero_crossings(x_yetvir[n0:n1], pad=False)



print("No. of Zero Crossings :")

print('For Ameavo :', sum(zero_crossings_ameavo))

print('For Mouchi :', sum(zero_crossings_mouchi))

print('For Nutwoo :', sum(zero_crossings_nutwoo))

print('For Pygnut :', sum(zero_crossings_pygnut))

print('For Vesspa :', sum(zero_crossings_vesspa))

print('For Yetvir :', sum(zero_crossings_yetvir))
fs=15

mfccs_ameavo = librosa.feature.mfcc(x_ameavo, sr=fs)

mfccs_mouchi = librosa.feature.mfcc(x_mouchi, sr=fs)

mfccs_nutwoo = librosa.feature.mfcc(x_nutwoo, sr=fs)

mfccs_pygnut = librosa.feature.mfcc(x_pygnut, sr=fs)

mfccs_vesspa = librosa.feature.mfcc(x_vesspa, sr=fs)

mfccs_yetvir = librosa.feature.mfcc(x_yetvir, sr=fs)



print("MFCC shapes :")

print("Ameavo :", mfccs_ameavo.shape)

print("Mouchi :", mfccs_mouchi.shape)

print("Nutwoo :", mfccs_nutwoo.shape)

print("Pygnut :", mfccs_pygnut.shape)

print("Vesspa :", mfccs_vesspa.shape)

print("Yetvir :", mfccs_yetvir.shape)



#Displaying  the MFCCs:

fig, ax = plt.subplots(6, figsize = (16, 14))

fig.suptitle('MFCCs Display', fontsize=14)

librosa.display.specshow(mfccs_ameavo, sr=sr, x_axis='time', cmap='spring', ax=ax[0])

ax[0].set_ylabel('Ameavo')

librosa.display.specshow(mfccs_mouchi, sr=sr, x_axis='time', cmap='spring', ax=ax[1])

ax[1].set_ylabel('Mouchi')

librosa.display.specshow(mfccs_nutwoo, sr=sr, x_axis='time', cmap='spring', ax=ax[2])

ax[2].set_ylabel('Nutwoo')

librosa.display.specshow(mfccs_pygnut, sr=sr, x_axis='time', cmap='spring', ax=ax[3])

ax[3].set_ylabel('Pygnut')

librosa.display.specshow(mfccs_vesspa, sr=sr, x_axis='time', cmap='spring', ax=ax[4])

ax[4].set_ylabel('Vesspa')

librosa.display.specshow(mfccs_yetvir, sr=sr, x_axis='time', cmap='spring', ax=ax[5])

ax[5].set_ylabel('Yetvir')
hop_length=12

chromagram_ameavo = librosa.feature.chroma_stft(x_ameavo, sr=sr, hop_length=hop_length)

chromagram_mouchi = librosa.feature.chroma_stft(x_mouchi, sr=sr, hop_length=hop_length)

chromagram_nutwoo = librosa.feature.chroma_stft(x_nutwoo, sr=sr, hop_length=hop_length)

chromagram_pygnut = librosa.feature.chroma_stft(x_pygnut, sr=sr, hop_length=hop_length)

chromagram_vesspa = librosa.feature.chroma_stft(x_vesspa, sr=sr, hop_length=hop_length)

chromagram_yetvir = librosa.feature.chroma_stft(x_yetvir, sr=sr, hop_length=hop_length)



fig, ax = plt.subplots(6, figsize = (16, 14))

fig.suptitle('Chromagrams Display', fontsize=14)

librosa.display.specshow(chromagram_ameavo, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='spring', ax=ax[0])

ax[0].set_ylabel('Ameavo')

librosa.display.specshow(chromagram_mouchi, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='spring', ax=ax[1])

ax[1].set_ylabel('Mouchi')

librosa.display.specshow(chromagram_nutwoo, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='spring', ax=ax[2])

ax[2].set_ylabel('Nutwoo')

librosa.display.specshow(chromagram_pygnut, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='spring', ax=ax[3])

ax[3].set_ylabel('Pygnut')

librosa.display.specshow(chromagram_vesspa, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='spring', ax=ax[4])

ax[4].set_ylabel('Vesspa')

librosa.display.specshow(chromagram_yetvir, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='spring', ax=ax[5])

ax[5].set_ylabel('Yetvir')
import cv2

import audioread

import logging

import os

import random

import time

import warnings



import librosa

import numpy as np

import pandas as pd

import soundfile as sf

import torch

import torch.nn as nn

import torch.cuda

import torch.nn.functional as F

import torch.utils.data as data



from contextlib import contextmanager

from pathlib import Path

from typing import Optional



from fastprogress import progress_bar

from sklearn.metrics import f1_score

from torchvision import models
def set_seed(seed: int = 12345):

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)  # type: ignore

    torch.backends.cudnn.deterministic = True  # type: ignore

    torch.backends.cudnn.benchmark = True  # type: ignore

    

    

def get_logger(out_file=None):

    logger = logging.getLogger()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger.handlers = []

    logger.setLevel(logging.INFO)



    handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    handler.setLevel(logging.INFO)

    logger.addHandler(handler)



    if out_file is not None:

        fh = logging.FileHandler(out_file)

        fh.setFormatter(formatter)

        fh.setLevel(logging.INFO)

        logger.addHandler(fh)

    logger.info("logger set up")

    return logger

    

    

@contextmanager

def timer(name: str, logger: Optional[logging.Logger] = None):

    t0 = time.time()

    msg = f"[{name}] start"

    if logger is None:

        print(msg)

    else:

        logger.info(msg)

    yield



    msg = f"[{name}] done in {time.time() - t0:.2f} s"

    if logger is None:

        print(msg)

    else:

        logger.info(msg)
logger = get_logger("main.log")

set_seed(12345)
TARGET_SR = 36000

TEST = Path("../input/birdsong-recognition/test_audio").exists()
if TEST:

    DATA_DIR = Path("../input/birdsong-recognition/")

else:

    # dataset created by @shonenkov, thanks!

    DATA_DIR = Path("../input/birdcall-check/")

    



test = pd.read_csv(DATA_DIR / "test.csv")

test_audio = DATA_DIR / "test_audio"





test.head()
sub = pd.read_csv("../input/birdsong-recognition/sample_submission.csv")

sub.to_csv("submission.csv", index=False)  #will be overwritten
class ResNet(nn.Module):

    def __init__(self, base_model_name: str, pretrained=False,

                 num_classes=264):

        super().__init__()

        base_model = models.__getattribute__(base_model_name)(

            pretrained=pretrained)

        layers = list(base_model.children())[:-2]

        layers.append(nn.AdaptiveMaxPool2d(1))

        self.encoder = nn.Sequential(*layers)



        in_features = base_model.fc.in_features



        self.classifier = nn.Sequential(

            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.25),

            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.25),

            nn.Linear(1024, num_classes))



    def forward(self, x):

        batch_size = x.size(0)

        x = self.encoder(x).view(batch_size, -1)

        x = self.classifier(x)

        multiclass_proba = F.softmax(x, dim=1)

        multilabel_proba = F.sigmoid(x)

        return {

            "logits": x,

            "multiclass_proba": multiclass_proba,

            "multilabel_proba": multilabel_proba

        }
model_config = {

    "base_model_name": "resnet50",

    "pretrained": False,

    "num_classes": 264

}



melspectrogram_parameters = {

    "n_mels": 128,

    "fmin": 20,

    "fmax": 20000

}



weights_path = "../input/birdcall-resnet50-init-weights/best.pth"
BIRD_CODE = {

    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,

    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,

    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,

    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,

    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,

    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,

    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,

    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,

    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,

    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,

    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,

    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,

    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,

    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,

    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,

    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,

    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,

    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,

    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,

    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,

    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,

    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,

    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,

    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,

    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,

    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,

    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,

    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,

    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,

    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,

    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,

    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,

    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,

    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,

    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,

    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,

    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,

    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,

    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,

    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,

    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,

    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,

    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,

    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,

    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,

    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,

    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,

    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,

    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,

    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,

    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,

    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,

    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263

}



INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}
def mono_to_color(X: np.ndarray,

                  mean=None,

                  std=None,

                  norm_max=None,

                  norm_min=None,

                  eps=1e-7):

    """

    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data

    """

    # Stack X as [X,X,X]

    X = np.stack([X, X, X], axis=-1)



    # Standardize

    mean = mean or X.mean()

    X = X - mean

    std = std or X.std()

    Xstd = X / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Normalize to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V





class TestDataset(data.Dataset):

    def __init__(self, df: pd.DataFrame, clip: np.ndarray,

                 img_size=224, melspectrogram_parameters={}):

        self.df = df

        self.clip = clip

        self.img_size = img_size

        self.melspectrogram_parameters = melspectrogram_parameters

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx: int):

        SR = 36000

        sample = self.df.loc[idx, :]

        site = sample.site

        row_id = sample.row_id

        

        if site == "site_3":

            y = self.clip.astype(np.float32)

            len_y = len(y)

            start = 0

            end = SR * 5

            images = []

            while len_y > start:

                y_batch = y[start:end].astype(np.float32)

                if len(y_batch) != (SR * 5):

                    break

                start = end

                end = end + SR * 5

                

                melspec = librosa.feature.melspectrogram(y_batch,

                                                         sr=SR,

                                                         **self.melspectrogram_parameters)

                melspec = librosa.power_to_db(melspec).astype(np.float32)

                image = mono_to_color(melspec)

                height, width, _ = image.shape

                image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))

                image = np.moveaxis(image, 2, 0)

                image = (image / 255.0).astype(np.float32)

                images.append(image)

            images = np.asarray(images)

            return images, row_id, site

        else:

            end_seconds = int(sample.seconds)

            start_seconds = int(end_seconds - 5)

            

            start_index = SR * start_seconds

            end_index = SR * end_seconds

            

            y = self.clip[start_index:end_index].astype(np.float32)



            melspec = librosa.feature.melspectrogram(y, sr=SR, **self.melspectrogram_parameters)

            melspec = librosa.power_to_db(melspec).astype(np.float32)



            image = mono_to_color(melspec)

            height, width, _ = image.shape

            image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))

            image = np.moveaxis(image, 2, 0)

            image = (image / 255.0).astype(np.float32)



            return image, row_id, site
def get_model(config: dict, weights_path: str):

    model = ResNet(**config)

    checkpoint = torch.load(weights_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda")

    model.to(device)

    model.eval()

    return model
def prediction_for_clip(test_df: pd.DataFrame, 

                        clip: np.ndarray, 

                        model: ResNet, 

                        mel_params: dict, 

                        threshold=0.60):



    dataset = TestDataset(df=test_df, 

                          clip=clip,

                          img_size=224,

                          melspectrogram_parameters=mel_params)

    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    model.eval()

    prediction_dict = {}

    for image, row_id, site in progress_bar(loader):

        site = site[0]

        row_id = row_id[0]

        if site in {"site_1", "site_2"}:

            image = image.to(device)



            with torch.no_grad():

                prediction = model(image)

                proba = prediction["multilabel_proba"].detach().cpu().numpy().reshape(-1)



            events = proba >= threshold

            labels = np.argwhere(events).reshape(-1).tolist()



        else:

            # to avoid prediction on large batch

            image = image.squeeze(0)

            batch_size = 16

            whole_size = image.size(0)

            if whole_size % batch_size == 0:

                n_iter = whole_size // batch_size

            else:

                n_iter = whole_size // batch_size + 1

                

            all_events = set()

            for batch_i in range(n_iter):

                batch = image[batch_i * batch_size:(batch_i + 1) * batch_size]

                if batch.ndim == 3:

                    batch = batch.unsqueeze(0)



                batch = batch.to(device)

                with torch.no_grad():

                    prediction = model(batch)

                    proba = prediction["multilabel_proba"].detach().cpu().numpy()

                    

                events = proba >= threshold

                for i in range(len(events)):

                    event = events[i, :]

                    labels = np.argwhere(event).reshape(-1).tolist()

                    for label in labels:

                        all_events.add(label)

                        

            labels = list(all_events)

        if len(labels) == 0:

            prediction_dict[row_id] = "nocall"

        else:

            labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))

            label_string = " ".join(labels_str_list)

            prediction_dict[row_id] = label_string

    return prediction_dict
def prediction(test_df: pd.DataFrame,

               test_audio: Path,

               model_config: dict,

               mel_params: dict,

               weights_path: str,

               threshold = 0.75):

    model = get_model(model_config, weights_path)

    unique_audio_id = test_df.audio_id.unique()



    warnings.filterwarnings("ignore")

    prediction_dfs = []

    for audio_id in unique_audio_id:

        with timer(f"Loading {audio_id}", logger):

            clip, _ = librosa.load(test_audio / (audio_id + ".mp3"),

                                   sr=TARGET_SR,

                                   mono=True,

                                   res_type="kaiser_fast")

        

        test_df_for_audio_id = test_df.query(

            f"audio_id == '{audio_id}'").reset_index(drop=True)

        with timer(f"Prediction on {audio_id}", logger):

            prediction_dict = prediction_for_clip(test_df_for_audio_id,

                                                  clip=clip,

                                                  model=model,

                                                  mel_params=mel_params,

                                                  threshold=threshold)

        row_id = list(prediction_dict.keys())

        birds = list(prediction_dict.values())

        prediction_df = pd.DataFrame({

            "row_id": row_id,

            "birds": birds

        })

        prediction_dfs.append(prediction_df)

    

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)

    return prediction_df
submission = prediction(test_df=test,

                        test_audio=test_audio,

                        model_config=model_config,

                        mel_params=melspectrogram_parameters,

                        weights_path=weights_path,

                        threshold=0.99)

submission.to_csv("submission.csv", index=False)
print("submission successful")