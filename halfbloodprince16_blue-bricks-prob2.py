import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

#step 1-> Use webrtc to make a triparty video conference.

#step 2-> Save the conference video at a local system disk.

#step 3-> Now convert the video file to audio file and extract text out of it to do sentiment analysis over it.

#step 4-> As mentioned to use spacy.io API to classify the sentiments but rather i would use pralleldots 

#         API to do same.

#step 5-> Relax we r done.
!pip install SpeechRecognition

!pip install paralleldots

!pip install moviepy
import moviepy.editor as mp

clip = mp.VideoFileClip("conf.mp4").subclip(0,20)

clip.audio.write_audiofile("audio.wav")
import speech_recognition as sr 

r = sr.Recognizer() 

file = sr.AudioFile('../input/audio.wav')



with file as source:

    audio = r.record(source)



text = r.recognize_google(audio)
print(text)
import paralleldots

paralleldots.set_api_key("Q6hg3AsjPpyejLgpkF041xQvGdvZm4UsrvFImWcfyLs")





# for single sentence

lang_code="en"

response=paralleldots.sentiment(text,lang_code)

print(response)