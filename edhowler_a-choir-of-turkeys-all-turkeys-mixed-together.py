#Let's load in some basics and make sure our files are all here
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
from __future__ import unicode_literals
import youtube_dl

# Downloading all training videos that are turkeys
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': u'%(id)s'
}
audios = []
for index, video in train.iterrows():
    if video['is_turkey'] == 1:
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(['http://www.youtube.com/watch?v=' + video['vid_id']])
            audios.append(video)
        except:
            # Some videos aren't available anymore
            pass

from moviepy.editor import *
from random import randint

# Clipping and gathering all turkey parts from each video
all_clips = []
for audio in audios:
    try:
        clip = AudioFileClip(audio['vid_id'])
        end_time = audio['end_time_seconds_youtube_clip']
        if clip.duration <= audio['end_time_seconds_youtube_clip']:
            end_time = None
        clip = AudioFileClip(audio['vid_id']).subclip(audio['start_time_seconds_youtube_clip'], end_time)
        # If the clip duration is less than 10 seconds, put it in a random place inside the 10 seconds limit
        all_clips.append(clip.set_start(randint(0, 10 - clip.duration)))
    except:
        pass
compositeAudioClip = CompositeAudioClip(all_clips)
# Generate mp3 file with all turkeys together
compositeAudioClip.write_audiofile("mix.mp3", 44100, progress_bar=False)
import IPython.display as ipd
ipd.Audio('mix.mp3')
# Audio with all turkeys together!
# Clipping and gathering all turkey parts from each video
all_clips = []
# Sorry, can't allocate memory for all clips in sequence
for audio in audios[:100]:
    try:
        clip = AudioFileClip(audio['vid_id'])
        end_time = audio['end_time_seconds_youtube_clip']
        if clip.duration <= audio['end_time_seconds_youtube_clip']:
            end_time = None
        clip = AudioFileClip(audio['vid_id']).subclip(audio['start_time_seconds_youtube_clip'], end_time)
        # If the clip duration is less than 10 seconds, put it in a random place inside the 10 seconds limit
        all_clips.append(clip)
    except:
        pass
compositeAudioClip = concatenate_audioclips(all_clips)
# Generate mp3 file with all turkeys, one each time
compositeAudioClip.write_audiofile("mix2.mp3", 44100, progress_bar=False)
import IPython.display as ipd
ipd.Audio('mix2.mp3')
# Audio with all turkeys in sequence!
test_data = [k for k in test['audio_embedding']]
submission = model.predict_classes(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})