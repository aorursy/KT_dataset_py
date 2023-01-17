# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import speech_recognition as sr

from time import ctime

import time

import os

from gtts import gTTS

def speak(audioString):

    print(audioString)

    tts = gTTS(text=audioString, lang='en')

    tts.save("audio.mp3")

    os.system("mpg321 audio.mp3")

def recordAudio():

    # Record Audio

    r = sr.Recognizer()

    with sr.Microphone() as source:

     print("Say something!")

    audio = r.listen(source)



    # Speech recognition using Google Speech Recognition

    data = ""

    try:

        # Uses the default API key

        # To use another API key: `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`

        data = r.recognize_google(audio)

        print("You said: " + data)

    except sr.UnknownValueError:

        print("Google Speech Recognition could not understand audio")

    except sr.RequestError as e:

        print("Could not request results from Google Speech Recognition service; {0}".format(e))



    return data
def jarvis(data):

    if "how are you" in data:

        speak("I am fine")



    if "what time is it" in data:

        speak(ctime())



    if "where is" in data:

        data = data.split(" ")

        location = data[2]

        speak("Hold on Frank, I will show you where " + location + " is.")

        os.system("chromium-browser https://www.google.nl/maps/place/" + location + "/&amp;")



# initialization

time.sleep(2)

speak("Hi Frank, what can I do for you?")

while 1:

    data = recordAudio()

    jarvis(data)