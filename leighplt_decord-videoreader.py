!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh 
import os, sys

sys.path.insert(0,'/kaggle/working/reader/python')



from decord import VideoReader

from decord import cpu, gpu

from decord.bridge import set_bridge

import glob
set_bridge('torch')
filenames = glob.glob('../input/deepfake-detection-challenge/test_videos/*.mp4')
%%time

## Be carefull GPU memory leak

shapes = []

for filename in filenames:

    video = VideoReader(filename, ctx=gpu())

    data = video.get_batch(range(len(video)))

    shapes += [data.size()]

    del video, data
%%time

## slower but stable

shapes = []

for filename in filenames:

    video = VideoReader(filename, ctx=cpu())

    data = video.get_batch(range(len(video)))

    shapes += [data.size()]

    del video, data
!rm -r reader && rm install.sh
!cp /kaggle/input/deepfake-detection-challenge/sample_submission.csv ./submission.csv