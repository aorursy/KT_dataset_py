!pip install pyttsx3
!pip install python-docx
#Covert Text from a file entered by user to speech
#install pyttsx3
import pyttsx3
#install python-docx
import docx
from pathlib import Path
engine = pyttsx3.init() # object creation

""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
print(rate)                        #printing current voice rate
engine.setProperty('rate', 150)     # setting up new voice rate

"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print (volume)                          #printing current volume level
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[1].id)   #changing index, changes voices (0 for male, 1 for female)

file = input("Enter file name (supports only .txt & .docx files): ")  #request user to enter file name

if Path(file).suffix == ".txt":
    infile = open(file, "r")
    # loop over each line, then print and say their text
    for i in infile.readlines():
        theText = i
        print(theText)
        engine.say(theText)
        engine.runAndWait()
        engine.stop()
elif Path(file).suffix == ".docx":
    infile = docx.Document(file)
    #loop over each Paragraph, then print and say their text
    for i in infile.paragraphs:
        theText = i.text
        print(theText)
        engine.say(theText)
        engine.runAndWait()
        engine.stop()
else:
    print("Document format is not supported") #print out error if unsupported file format is passed
