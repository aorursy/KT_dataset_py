%matplotlib inline

from memory_profiler import memory_usage

import os

import pandas as pd

from glob import glob

import numpy as np
%%capture

!apt-get install libav-tools -y
from fastai.vision import *

import librosa

import librosa.display

import pylab

import matplotlib

import gc
!mkdir /kaggle/working/train

!mkdir /kaggle/working/test
def create_spectrogram(filename,name):

    plt.interactive(False)

    clip, sample_rate = librosa.load(filename, sr=None)

    fig = plt.figure(figsize=[0.72,0.72])

    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    filename  = Path('/kaggle/working/train/' + name + '.jpg')

    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close()    

    fig.clf()

    plt.close(fig)

    plt.close('all')

    del filename,name,clip,sample_rate,fig,ax,S
def create_spectrogram_test(filename,name):

    plt.interactive(False)

    clip, sample_rate = librosa.load(filename, sr=None)

    fig = plt.figure(figsize=[0.72,0.72])

    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    filename  = Path('/kaggle/working/test/' + name + '.jpg')

    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close()    

    fig.clf()

    plt.close(fig)

    plt.close('all')

    del filename,name,clip,sample_rate,fig,ax,S
Data_dir=np.array(glob("../input/train/Train/*"))
%load_ext memory_profiler
%%memit 

i=0

for file in Data_dir[i:i+1500]:

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_spectrogram(filename,name)
gc.collect()
%%memit 

i=1500

for file in Data_dir[i:i+1500]:

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_spectrogram(filename,name)
gc.collect()
%%memit 

i=3000

for file in Data_dir[i:i+1500]:

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_spectrogram(filename,name)
gc.collect()
%%memit 

i=4500

for file in Data_dir[i:]:

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_spectrogram(filename,name)
gc.collect()
path = Path('/kaggle/working/')

np.random.seed(42)

data = ImageDataBunch.from_csv(path,csv_labels='../input/train.csv', folder="train", valid_pct=0.2, suffix='.jpg',

        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
data.classes
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-4,1e-3))
learn.save('stage-2')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, max_lr=slice(1e-6,2e-6))
learn.save('stage-3')
Test_dir=np.array(glob("../input/test/Test/*"))
%%memit 

i=0

for file in Test_dir[i:i+1500]:

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_spectrogram_test(filename,name)
gc.collect()
%%memit 

i=1500

for file in Test_dir[i:]:

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_spectrogram_test(filename,name)
gc.collect()
learn.load('stage-3')

test_csv = pd.read_csv('../input/test.csv')
with open('output.csv',"w") as file:

    file.write("ID,Prediction\n")

    for test in test_csv.ID:

        img = open_image('/kaggle/working/test/'+str(test)+'.jpg')

        prediction = str(learn.predict(img)[0]).split()[0]

        file.write(str(test)+','+prediction)

        file.write('\n')
output = pd.read_csv('output.csv')

output.head()
%%capture

!apt-get install zip

!zip -r train.zip /kaggle/working/train/

!zip -r test.zip /kaggle/working/test/

!rm -rf train/*

!rm -rf test/*