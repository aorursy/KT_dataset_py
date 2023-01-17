!pip install pydub
import os

from os import path
from pydub import AudioSegment


 

test_set="./test_set/"
 

os.mkdir(test_set) 
 

for dirname, _, filenames in os.walk('../input/quran-asr-challenge/test_set'):
    for filename in filenames:
        # files                                                                         
        src = "../input/quran-asr-challenge/test_set/"+filename
        dst = test_set+os.path.splitext(filename)[0]+".wav"
        # convert wav to mp3                                                            
        sound = AudioSegment.from_mp3(src)
        sound = sound.set_frame_rate(8000)
        sound.export(dst, format="wav")
