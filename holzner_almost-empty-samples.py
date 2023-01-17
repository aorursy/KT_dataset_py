import os

import numpy as np

from scipy.io import wavfile



from tqdm import tqdm



import pandas as pd



import IPython.display as ipd



import re



%matplotlib inline

import matplotlib.pyplot as plt
audio_paths = [

    "../input/train/audio",

 #  "../input/test/audio",

]
wav_fnames = []



for audio_path in audio_paths:

    for dir_name, dirs, files in os.walk(audio_path):

                

        for fname in files:

            

            if fname.endswith(".wav"):

                wav_fnames.append(os.path.join(dir_name,fname))
max_diffs = []

speakers = []



with tqdm(total = len(wav_fnames), mininterval = 1, unit = "files") as progbar:



    for fname in wav_fnames:

        # read the .wav file

        sample_rate, samples = wavfile.read(fname)

        

        # convert to float in order to avoid underflow

        # when subtracting...

        samples = samples.astype('float32')

        

        max_diffs.append(max(samples) - min(samples))

        

        # extract speaker name

        mo = re.match("([0-9a-f]{8})_.*\.wav$", os.path.basename(fname))

        if mo:

            speakers.append(mo.group(1))

        else:

            speakers.append(None)

        

        progbar.update()



# create a Pandas dataframe

max_diffs = pd.DataFrame(dict(max_diff = max_diffs, 

                              fname = wav_fnames[:len(max_diffs)],

                              speaker = speakers))
max_diffs.sort_values('max_diff', ascending=True, inplace=True)

max_diffs.reset_index(inplace = True, drop = True);
max_diffs[0:50]
max_diffs[:462].to_csv("least-amplitudes.csv")
plt.figure(figsize = (12,10))



plt.subplot(1,2,1)

max_diffs['max_diff'][max_diffs['max_diff'] < 30].hist(bins = 100)

plt.xlabel('max difference')



plt.subplot(1,2,2)

max_diffs['max_diff'].hist(bins = 100)

plt.xlabel('max difference');
# noisy but understandable

# maxdiff 363, index 499

sample_rate, samples = wavfile.read("../input/train/audio/up/72198b96_nohash_0.wav")

ipd.Audio(samples, rate=sample_rate)
# very noisy

# maxdiff 10, index 24

sample_rate, samples = wavfile.read("../input/train/audio/go/712e4d58_nohash_4.wav")

ipd.Audio(samples, rate=sample_rate)
# very noisy (same speaker as before)

# maxdiff 11, index 29

sample_rate, samples = wavfile.read("../input/train/audio/left/712e4d58_nohash_2.wav")

ipd.Audio(samples, rate=sample_rate)
# very noisy (same speaker as before)

# maxdiff 9, index 8

sample_rate, samples = wavfile.read("../input/train/audio/eight/712e4d58_nohash_1.wav")

ipd.Audio(samples, rate=sample_rate)
# same speaker as before but clearly understandable

# maxdiff 4969, index 5791

sample_rate, samples = wavfile.read("../input/train/audio/zero/712e4d58_nohash_4.wav")

ipd.Audio(samples, rate=sample_rate)
# different speaker, noisy

# maxdiff 9, index 7

sample_rate, samples = wavfile.read("../input/train/audio/stop/7fd25f7c_nohash_1.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 11, index 29

sample_rate, samples = wavfile.read("../input/train/audio/left/712e4d58_nohash_2.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 16, index 51

sample_rate, samples = wavfile.read("../input/train/audio/right/e96a5020_nohash_3.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 25, index 100

sample_rate, samples = wavfile.read("../input/train/audio/off/e96a5020_nohash_1.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 29, index 113

sample_rate, samples = wavfile.read("../input/train/audio/up/ad63d93c_nohash_0.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 73, index 151

sample_rate, samples = wavfile.read("../input/train/audio/two/ced835d3_nohash_2.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 93, index 199

sample_rate, samples = wavfile.read("../input/train/audio/stop/ced835d3_nohash_0.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 106, index 251

sample_rate, samples = wavfile.read("../input/train/audio/five/ced835d3_nohash_4.wav")

ipd.Audio(samples, rate=sample_rate)
# noisy

# maxdiff 137, index 300

sample_rate, samples = wavfile.read("../input/train/audio/off/f8f60f59_nohash_1.wav")

ipd.Audio(samples, rate=sample_rate)
# unrecognizeable

# maxdiff 188, index 349

sample_rate, samples = wavfile.read("../input/train/audio/dog/fd395b74_nohash_0.wav")

ipd.Audio(samples, rate=sample_rate)
# unrecognizeable

# maxdiff 240, index 400

sample_rate, samples = wavfile.read("../input/train/audio/nine/fd395b74_nohash_1.wav")

ipd.Audio(samples, rate=sample_rate)
# unrecognizeable

# maxdiff 294, index 447

sample_rate, samples = wavfile.read("../input/train/audio/eight/742d6431_nohash_5.wav")

ipd.Audio(samples, rate=sample_rate)
# barely understandable

# maxdiff 309, index 456

sample_rate, samples = wavfile.read("../input/train/audio/three/fd395b74_nohash_0.wav")

ipd.Audio(samples, rate=sample_rate)
# clear

# maxdiff 314, index 462

sample_rate, samples = wavfile.read("../input/train/audio/go/ec5ab5d5_nohash_0.wav")

ipd.Audio(samples, rate=sample_rate)
# clear

# maxdiff 333, index 476

sample_rate, samples = wavfile.read("../input/train/audio/on/9799379a_nohash_0.wav")

ipd.Audio(samples, rate=sample_rate)
# perfectly understandable

# maxdiff 420, index 549

sample_rate, samples = wavfile.read("../input/train/audio/bed/26e9ae6b_nohash_1.wav")

ipd.Audio(samples, rate=sample_rate)
max_diffs[max_diffs.speaker == "712e4d58"]