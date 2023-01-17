# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# Referenced from https://www.kdnuggets.com/2020/06/easy-speech-text-python.html

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Installing Library
!pip install SpeechRecognition
# Importing Library
import speech_recognition as sr
# Initializing recognizer class (for recognizing the speech)
r = sr.Recognizer()
# Locating audio file
audio = '/kaggle/input/male.wav'
# Reading Audio file as source
# listening the audio file and store in audio_text variable
with sr.AudioFile(audio) as source:
    
    audio_text = r.listen(source)
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
try:
        
    # using google speech recognition
    text = r.recognize_google(audio_text)
    print('Converting audio transcripts into text ...')
    print(text)
     
except:
    print('Sorry.. run again...')
#If we want to read file of different language then we need to add that option  
#Adding french langauge option
#text = r.recognize_google(audio_text, language = "fr-FR")