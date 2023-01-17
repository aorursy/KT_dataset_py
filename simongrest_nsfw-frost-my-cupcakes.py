import os

from scipy.io import wavfile

import IPython.display as ipd
ipd.Audio(wavfile.read("../input/train_curated/7f409e1a.wav")[1], rate=44100)