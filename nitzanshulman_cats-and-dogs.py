# Install without printing to std

!pip install git+https://github.com/rbracco/fastai2_audio.git
from fastai2.torch_basics import *

from fastai2.basics import *

from fastai2.data.all import *

from fastai2.callback.all import *

from fastai2.vision.all import *



from fastai2_audio.core import *

from fastai2_audio.augment import *



import torchaudio
data_path = Path("/kaggle/input/audio-cats-and-dogs/cats_dogs/train")

test_data_path = Path("/kaggle/input/audio-cats-and-dogs/cats_dogs/test")

x = AudioGetter("", recurse=True, folders=None)

x(data_path)
#crop 2s from the signal and turn it to a MelSpectrogram with no augmentation

cfg_voice = AudioConfig.Voice()

a2s = AudioToSpec.from_cfg(cfg_voice)

crop_2000ms = CropSignal(2000)

tfms = [crop_2000ms, a2s]



auds = DataBlock(blocks=(AudioBlock, CategoryBlock),

                 get_items=get_audio_files, 

                 splitter=RandomSplitter(),

                 item_tfms = tfms,

                 get_y=lambda x: str(x).split('/')[-1].split('_')[0])
cats = [y for _,y in auds.datasets(data_path)]
a = auds.datasets(data_path)
#verify categories are being correctly assigned

test_eq(min(cats).item(), 0)

test_eq(max(cats).item(), 1)
dbunch = auds.dataloaders(data_path, bs=64)

def alter_learner(learn, channels=1):

    learn.model[0][0].in_channels=channels

    learn.model[0][0].weight = torch.nn.parameter.Parameter(learn.model[0][0].weight[:,1,:,:].unsqueeze(1))



learn = Learner(dbunch, 

                xresnet18(),

                torch.nn.CrossEntropyLoss(), 

                metrics=[accuracy])

nchannels = dbunch.one_batch()[0].shape[1]

alter_learner(learn, nchannels)
learn.lr_find()
learn.fit_one_cycle(25)
learn.save("95-precent")
learn.load("95-precent")
# Unrelated reading data with pytorch audio

import torch

import torchaudio

import matplotlib.pyplot as plt



def read_data(path):

    X = []

    Y = []

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            # Chk it is audio file

            if filename.endswith("wav"):

                # Label 0 is dogs

                label = 0

                

                if "cat" in filename:

                    # label 1 is cats

                    label = 1

                X.append(list(torchaudio.load(os.path.join(dirname,filename))))

                Y.append(label)

    

    return X, Y