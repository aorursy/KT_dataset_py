import librosa

import IPython.display as ipd
# Saeglópur from Sigur Rós

sample, sr = librosa.load(path='../input/train_noisy/9e09475c.wav', sr=None)

ipd.Audio(sample, rate=44100, autoplay=True) 
# あなただけ見つめてる from Slam dunk!

sample, sr = librosa.load(path='../input/train_noisy/0e29068e.wav', sr=None)

ipd.Audio(sample, rate=44100, autoplay=True)