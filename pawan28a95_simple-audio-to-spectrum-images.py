import matplotlib.pyplot as plt

# import pylab as plt

from matplotlib.pyplot import specgram

import librosa

import numpy as np

from IPython.display import Audio

import librosa

import librosa.display

import matplotlib.pyplot as plt

import pandas as pd

import os

from pathlib import Path

import shutil

import sys

import warnings

if not sys.warnoptions:

    warnings.simplefilter("ignore")

from tqdm import tqdm

import gc

from matplotlib import figure
os.listdir('../input/birdsong-recognition/')
def create_fold_spectrograms(audio_file_path,save_img_path ):

    audio, sr = librosa.load(audio_file_path, sr =None)

#     fig = figure.Figure(figsize=[0.72,0.72])

    fig = plt.figure(figsize=[0.72,0.72])

    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    ax.set_frame_on(False)

    filename  = Path(audio_file_path).name.replace('.mp3','.png').replace('.mp2','.png').replace('.aac','.png').replace('.wav','.png')

    S = librosa.feature.melspectrogram(y=audio, sr=sr)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    fig.savefig(os.path.join(save_img_path, filename), dpi=None, bbox_inches='tight',pad_inches=0)

    fig.clear()

    ax.cla()

    plt.close(fig)

    plt.close()

    plt.close('all')

    plt.cla()

    fig.clf()

    plt.clf()

    plt.close()

    gc.collect()

    del S

    ### Audio Sample

#     Audio(audio, rate =sr)
os.mkdir('train')
trainDir = Path('../input/birdsong-recognition/train_audio/')

imgFolders = ['amegfi','amepip','amered','amerob','amewig','amtspa','annhum','astfly','baleag','barswa','belkin1','bewwre','bkbcuc','bkbmag1','bkpwar','bktspa','blujay','bongul','brdowl','brespa','brncre','brnthr','btywar','bushti','calgul','calqua','camwar','cangoo','casfin','casvir','cedwax','chispa','chiswi','chukar','comgol','comloo','commer','comnig','comred','comter','cowscj1','doccor','dusfly','easkin','easmea','eucdov','fiespa','fiscro','foxspa','gadwal','gcrfin','gnttow','gnwtea','gockin','gocspa','grcfly','greegr','greroa','grhowl','grnher','grycat','hamfly','hergul','horgre','houfin','juntit1','labwoo','larspa','leabit','lecthr','lesgol','lesyel','lewwoo','linspa','lobdow','louwat','mallar3','moublu','mouchi','moudov','norcar','normoc','norpin','nrwswa','olsfly','orcwar','ovenbi1','pasfly','perfal','phaino','pilwoo','pinsis','plsvir','pygnut','rebnut','redcro','redhea','renpha','reshaw','rethaw','rthhum','rudduc','rufgro','sagspa1','sagthr','savspa','saypho','scatan','scoori','sheowl','snobun','sonspa','swahaw','swaspa','swathr','tuftit','tunswa','veery','vigswa','wessan','westan','whfibi','whtswi','wilsni1','wooscj2','woothr','y00475','yebfly','yebsap','yelwar','yerwar']

len(imgFolders)
%time

trainDir = Path('../input/birdsong-recognition/train_audio/')

for ebird_code in imgFolders:

    gc.collect()

    print(ebird_code)

    if not os.path.exists(os.path.join('train',ebird_code)):

        os.makedirs(os.path.join('train',ebird_code))

    for audio in os.listdir(Path(trainDir/ebird_code)):

        print(audio)

        create_fold_spectrograms(audio_file_path = Path(trainDir/ebird_code/audio),\

                                 save_img_path = os.path.join('/kaggle/working/train',ebird_code))

        gc.collect()
shutil.make_archive('train_zipped', 'zip', 'train')