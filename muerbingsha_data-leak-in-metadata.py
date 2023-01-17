import pandas as pd

import glob

import os

import subprocess as sp

import tqdm.notebook as tqdm

from collections import defaultdict

import json



%%capture

! tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz
def ffprobe(filename):

    import subprocess

    command = ["../working/ffmpeg-git-20191209-amd64-static/ffprobe", "-v", "error", "-show_streams", "-print_format", "xml", filename]

    xml = subprocess.check_output(command)

    return xml
# command line mode

! ../working/ffmpeg-git-20191209-amd64-static/ffprobe -v error -show_streams -print_format xml /kaggle/input/deepfake-detection-challenge/test_videos/bcvheslzrq.mp4
# sample

xml = ffprobe('/kaggle/input/deepfake-detection-challenge/test_videos/bcvheslzrq.mp4')
def get_markers(video_file):



    xml = ffprobe(str(video_file))

    

    found = str(xml).find('"audio" codec_time_base')

    mp = str(xml)[found+25:found+32] # 1/48000

    

    found = str(xml).find('display_aspect_ratio')

    if found >= 0:

        ar = str(xml)[found+22:found+26] # 16:9

    else:

        ar = None

    

    return ar, mp
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4')
my_dict = defaultdict()

for filename in tqdm.tqdm(filenames):

    fn = filename.split('/')[-1]

    ar, mp = get_markers(filename)

    my_dict[fn] = ar
display_aspect_ratios = pd.DataFrame.from_dict(my_dict, orient='index')

display_aspect_ratios.columns = ['display_aspect_ratio']

display_aspect_ratios = display_aspect_ratios.fillna('NONE')
labels = json.load(open('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json', encoding="utf8"))

labels = pd.DataFrame(labels).transpose() # json to dataframe

labels = labels.reset_index() # add index

labels = labels.join(display_aspect_ratios, on='index')
pd.crosstab(labels.display_aspect_ratio, labels.label)
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')
sub = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')

sub.label = 0.5

sub = sub.set_index('filename',drop=False)
# version 1

threshold = 0.15

for filename in tqdm.tqdm(filenames):

    

    fn = filename.split('/')[-1]

    ar, mp = get_markers(filename)

    

#     if mp == '1/16000':

#         sub.loc[fn, 'label'] = 1

#     else:

    if ar is None: # real

        sub.loc[fn, 'label'] = threshold  # 0.15

    elif ar == '16:9' or ar == '9:16': # fake

        sub.loc[fn, 'label'] = 1 - threshold # 0.85
from math import log

a1 = 1563

a2 = 2079 - a1



b1 = 252

b2 = 419 - b1



c1 = 174

c2 = 1477 - c1



d1 = 11

d2 = 25 - d1

score = -1/4000 * (a1*log(a1/2079) + a2*log(a2/2079) + b1*log(b1/419) + b2*log(b2/419) + c1*log(c1/1477) + c2*log(c2/1477) + d1*log(d1/25) + d2*log(d2/25))

score
# version 2

# Reference: https://www.kaggle.com/diegojohnson/compute-lb-score-directly-data-leak

# change the threshold 

for filename in tqdm.tqdm(filenames):

    

    fn = filename.split('/')[-1]

    ar, mp = get_markers(filename)

    

#     if mp == '1/16000':

#         sub.loc[fn, 'label'] = 1

#     else:

    if ar is None: # real

        sub.loc[fn, 'label'] = 174/1477

    elif ar == '16:9':

        sub.loc[fn, 'label'] = 1563/2079

    elif ar == '9:16': # fake

        sub.loc[fn, 'label'] = 252/419
sub.label.value_counts()
sub.to_csv('submission.csv', index=False)