# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import re
import subprocess
import cv2
import matplotlib.pyplot as plt
from ecapture import ecapture as ec


import speech_recognition as sr
import os
import sys
import re
import webbrowser
import smtplib
import requests
import subprocess
from pyowm import OWM
from urllib.request import urlopen
from socket import timeout
#import youtube_dl
#import vlc
#import urllib
#import urllib2
import json
#from bs4 import BeautifulSoup as soup
#from urllib2 import urlopen
#import wikipedia
import random
from time import strftime
import ctypes
import time
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
print(voices[1].id)
engine.setProperty('voice',voices[0].id)


def speek(audio):
    engine.say(audio)
    engine.runAndWait()
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speek("good morning!,sir" )
        
    elif hour>=12 and hour<18:
        speek("good afternoon sir")
        
    else:
        speek("good evening!, sir")
        
    speek("Hello i am Pointer. please tell me how may i help you!")
def takeCommand():
    
    r = sr.Recognizer()
    m = sr.Microphone()
    
    print("A moment of silence, please...")
    with m as source:r.adjust_for_ambient_noise(source)
    print("Say something!")
    with m as source:audio = r.listen(source)
    print("Got it! Now to recognize it...")
    value = ""
    
   # r = sr.Recognizer()
   # with sr.Microphone() as source:
   #     print("Listening...")
   #     r.pause_threshold = 1
   #     audio = r.listen(source)

    try:
        print("Recognizing...")    
        value = r.recognize_google(audio, language='en-in')
        print(f"User said: {value}\n")

    except Exception as e:
        # print(e)    
        print("Say that again please...")  
        return "None"
    return value
def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('yourgmailid@gmail.com', 'your_password')
    server.sendmail('yourgmailid@gmail.com', to, content)
    server.close()
def note(text):
    date = datetime.datetime.now()
    file_name = str(date).replace(":", "-") + "-note.txt"
    with open(file_name, "w") as f:
        f.write(text)

    subprocess.Popen(["notepad.exe", file_name])
def Find_location(querry):
    if "find location" in querry:           
        speek("what is the location")
        location = takeCommand().lower()
        url = 'https://google.nl/maps/place/' + location + '/&amp;'
        webbrowser.get().open(url)
        speek("Hold on , I will show you the location of " + location + " is.")
if __name__ =="__main__":
    NOTE_STRS = ["make a note", "write this down", "remember this"]
   # loc = ["find location", "find the location","pointer find the location","get me the location","check location","check the location"]
    code = ["open code","I need to update my code","open visual studio"]
    #speek("himanshu, is a boy")
    wishMe()
    #querry = takeCommand().lower()
    #respond(querry)
    
    
    #while True:
    if 1:
        
            querry = takeCommand().lower()
            print(querry)
            
            if "wikipedia" in querry:
                speek("Searching Wikipedia... please wait!")
                querry = querry.replace("wikipedia","")
                result = wikipedia.summary(querry, sentences=2)
                speek("According to wikipedia")
                print(result)
                speek(result)
                
            elif "who are you" in querry:
                speek("hello i am pointer not like a dangling pointer, i am your assistance")
            
            
            elif "what\ 's up" in querry:
                speek("just doing my thing")
                
            
            elif "hey" in querry or "hello" in querry:
                speek("hello sir")
                
            
            elif "open youtube" in querry:
                speek("opening youtube sir")
                webbrowser.open("youtube.com")
                
            
            elif "open google" in querry:
                speek("opening your dream google sir")
                webbrowser.open("google.com")
                
            
            elif 'how are you' in querry: 
                speek("I am fine, Thank you") 
                speek("How are you, Sir")
                
            
            elif "who i am" in querry: 
                speek("you are my boss himanshu, basically you are from rajasthan")
                
            
            elif "who made you" in querry or "who created you" in querry:  
                speek("I have been created by himanshu.")
                
            
            elif "what's your name" in querry or "What is your name" in querry: 
                speek("My friends call me pointer")
                
            elif "play music" in querry:
                music_dir = ""
                songs = os.listdir("your music dir path")
                print(songs)
                os.startfile(os.path.join(music_dir,songs[0]))
                
            elif "the time" in querry:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                speek(f"sir, the time is {strTime}")
            
            
            elif querry:  
                Find_location(querry)
                
                #speek("what is the location")
                #location = takeCommand().lower()
                #url = 'https://google.nl/maps/place/' + location + '/&amp;'
                #webbrowser.get().open(url)
                #speek("Hold on , I will show you the location of " + location + " is.") 
            
                
            elif querry:
                for ide in code:
                    if ide in querry:
                        CodePath = "your IDE path"
                        os.startfile(CodePath)
                        #break
                        speek("opening visual studio code, i know you love coding, but only for few time hahaha")
        
            elif "email to me" in querry:
                try:
                    speek("what should i say!")
                    content = takeCommand()
                    to = "senderid@gmail.com."
                    sendEmail(to, content)
                    speek("email has been sent")
                    
                except Exception as e:
                    print(e)
                    speek("sorry my boss. i am not able to send email")
                    
                    
                    
                    
            elif 'open' in querry:
                reg_ex = re.search('open (.+)', querry)
                if reg_ex:
                    domain = reg_ex.group(1)
                    print(domain)
                    url = 'https://www.' + domain
                    webbrowser.open(url)
                    speek('The website you have requested has been opened for you Sir.')
                else:
                    pass
                
                
            elif 'launch' in querry:
                reg_ex = re.search('launch (.*)', querry)
                if reg_ex:
                    appname = reg_ex.group(1)
                    appname1 = appname+".app"
                    subprocess.Popen(["open", "-n", "/Applications/" + appname1], stdout=subprocess.PIPE)
                    speek('I have launched the desired application')
                                
                    
            elif 'current weather' in querry:
                reg_ex = re.search('current weather in (.*)', querry)
                if reg_ex:
                     city = reg_ex.group(1)
                     owm = OWM(API_key='91cfde31bb1956791c4b1ac80abd4784')
                     obs = owm.weather_at_place('bangalore')
                     w = obs.get_weather()
                     k = w.get_status()
                     x = w.get_temperature(unit='celsius')
                     speek('Current weather in %s is %s. The maximum temperature is %0.2f and the minimum temperature is %0.2f degree celcius' % (city, k, x['temp_max'], x['temp_min']))
                    
                
                
            elif 'exit pointer' in querry:
                 speek('Bye bye Sir. Have a nice day')
                 sys.exit()
                 
            elif "camera" in querry:
                speek("opening camera sir")
                cap = cv2.VideoCapture(0)
                cap.set(3,480) 
                cap.set(4,360) 
                while True:
                	ret,frame = cap.read()
                    #cv2.imshow('frame', frame)
                	plt.show()
                	if cv2.waitKey(1) == ord('q'):
                		break
                
                cap.release()
                cv2.destroyAllWindows() 
                
            elif 'lock window' in querry: 
                    speek("locking the device") 
                    ctypes.windll.user32.LockWorkStation()
                    
                
            elif "camera" in querry or "take a photo" in querry: 
                ec.capture(0, "pointer Camera ", "img.jpg") 
                
            elif "don't listen" in querry or "stop listening" in querry: 
                speek("for how much time you want to stop pointer from listening commands") 
                a = int(takeCommand()) 
                time.sleep(a) 
                print(a)
                
                       
                
            elif querry:
                for phrase in NOTE_STRS:
                    if phrase in querry:
                        speek("What would you like me to write down?")
                        note_text = takeCommand().lower()
                        note(note_text)
                        speek("I've made a note of that.")
                        
            elif 'news' in querry: 
                  
                try:  
                    jsonObj = urlopen('''https://newsapi.org / v1 / articles?source = the-times-of-india&sortBy = top&apiKey =\\times of India Api key\\''') 
                    data = json.load(jsonObj) 
                    i = 1
                      
                    speek('here are some top news from the times of india') 
                    print('''=============== TIMES OF INDIA ============'''+ '\n') 
                      
                    for item in data['articles']: 
                          
                        print(str(i) + '. ' + item['title'] + '\n') 
                        print(item['description'] + '\n') 
                        speek(str(i) + '. ' + item['title'] + '\n') 
                        i += 1
                except Exception as e: 
                      
                    print(str(e)) 