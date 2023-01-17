# read in all the modules we'll use
# included by default
import numpy as np
import matplotlib.pyplot as plt

# added to this kernel
# pip install : SpeechRecognition
import speech_recognition as sr
# pip install: soundfile
import soundfile as sf
### Using speech recognition to transcribe speech

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile("../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/bed/1001.wav", ) as source:
    audio = r.record(source)  # read the entire audio file
    
# recognize speech using Google Speech Recognition
# IMPORTANT NOTE: You don't want to do this too many  times because the 
# default API key is rate limited & you'll hit it pretty quick
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
### create a fucntion to speed this process up

# our function takes in a file path & uses the default API key to 
# recognize what's in it. 

# IMPORTANT NOTE: You don't want to do this too many  times because the 
# default API key is rate limited & you'll hit it pretty quick
def recognize_speech_goog_default(file_path):
    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(file_path, ) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("Google Speech Recognition thinksteach you said " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

# use function to recognize the same file
recognize_speech_goog_default("../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/bed/1001.wav")
### Visualizing a spectrogram

# load in the data & samplerate
data, samplerate = sf.read("../input/synthetic-speech-commands-dataset/augmented_dataset_verynoisy/augmented_dataset_verynoisy/bed/1001.wav")

# plot a spectrogram
Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)

# add axis labels
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
# Now let's write a function to produce a spectrogram
# that looks like this with just the file path

# What do we do first?
def viz_spectrogram(file_path):
    # load in the data & samplerate
    data, samplerate = sf.read(file_path)

    # plot a spectrogram
    Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)

    # add axis labels
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    
viz_spectrogram("../input/synthetic-speech-commands-dataset/augmented_dataset_verynoisy/augmented_dataset_verynoisy/bed/1001.wav")
## First, write (or copy and paste) a chunk of code that you've ended up typing more than once

"strting" + " " + "asdfasdc"
## Now, write a function that will make it easier for you to use that code in the future
def concat_strings(string1, string2):
    new_string = string1 + " " + string2
    return new_string

concat_strings("adsfs","asfdsaf")
# Your sample code goes here


# Your revised sample code goes here


# Your code here :)


# revised code here :)

# import packages we'll need
import pandas as pd

# create a dataframe with random numbers in it
dataframe_random = pd.DataFrame(np.random.randint(low=0, high=10, size=(5,5)),
                    columns=['a', 'b', 'c', 'd', 'e'])

# Function to subtract the median from the mean for each column in a dataframe
# (a very simple measure of skew)
def simple_skew(df):
    means = df.mean(axis=None, skipna=None, level=None, numeric_only=None)
    medians = df.median(axis=None, skipna=None, level=None, numeric_only=None)

    return means - medians

# see if it works
simple_skew(dataframe_random)
# This function will let learners test their simple measures of skew
def test_simple_skew(function_name):
    # create a test dataset we know the correct answer for
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [10, 3.5, 6]})
    # store that answer (figure this out in another place & copy and
    # paste rather than writing the code to figure it out here so that 
    # students can't just copy & paste it. :)
    expected_output = pd.Series({'col1':0, 'col2':0.5})
    
    # check if the output of our function is the same as the expected
    # output & print helpful messages
    if expected_output.equals(function_name(df)):
        print("Correct! You've got the expected output.")
    else:
        print("Looks like that's not quite it.")
# use the function to test our work
test_simple_skew(simple_skew)
## Part 1: 

# Edit this function so that it returns a helpful message if students only 
# return the mean or the median

# This function will let students test themselves
def test_simple_skew(function_name):
    # create a test dataset we know the correct answer for
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [10, 3.5, 6]})
    # store that answer (figure this out in another place & copy and
    # paste rather than writing the code to figure it out here so that 
    # students can't just copy & paste it. :)
    expected_output = pd.Series({'col1':0, 'col2':0.5})
    
    mean = pd.Series({'col1':2, 'col2':6.5})
    # check if the output of our function is the same as the expected
    # output & print helpful messages
    if expected_output.equals(function_name(df)):
        print("Correct! You've got the expected output")
    elif mean.equals(function_name(df)):
        print("Close! Did you remember the median?")
    else:
        print("Looks like that's not quite it")
    # you can include additional helpful if statements here
    # to catch common errors & steer students in the right direction
    
    
## Part 2: 

# write a function that just returns the mean

def simple_skew_needs_improvement(df):
    means = df.mean(axis=None, skipna=None, level=None, numeric_only=None)
    medians = df.median(axis=None, skipna=None, level=None, numeric_only=None)

    return means


## Part 3:

# use your function from part 2 to test your test function from part 1

test_simple_skew(simple_skew_needs_improvement)