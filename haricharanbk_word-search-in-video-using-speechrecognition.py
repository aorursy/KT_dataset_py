!pip install moviepy

!pip install SpeechRecognition
from moviepy import editor         # for video and audio editing

from numpy import linspace         # for diving some interval into number of equal lengh parts.

import speech_recognition as sr    # actual speech recognition library

from math import ceil              # ceil function for finding ceil of some float value
class Trie:

    """

        Creates a Trie and adds words with corresponding time stamps into trie using make_trie function

    """

    def __init__(self):

        self.trie = dict()

        self.ans = dict()

        

        

    def make_trie(self, words:list, time:str) -> None:

        """

        Inserts the time stamp into the trie at all the words in the list of words

            words: words to insert into the trie

            time : time stamp associated with that word

        """

        for word in words:

            temp = self.trie

            for i in range(len(word)):

                temp = temp.setdefault(word[i], {})

            if '*' in temp:

                temp['*'].add(time)

            else:

                temp['*'] = {time}



    def find_similar_words(self, temp, letter, text):

        """

        An utility function for find_words

        """

        text += letter

        temp = temp[letter]

        remaining = list(temp)

        text_arr = [text] * len(remaining)

        for i in range(len(remaining)):

            if remaining[i] == '*':

                self.ans[text_arr[i]] = temp['*']

            else:

                self.find_similar_words(temp, remaining[i], text_arr[i])



    def find_words(self, word:str) -> None:

        """

        Takes a word as input and searches for its existance in trie. If exists, returns the time stamps associated.

        finally it returns all the similar words with that prefix and their time stamps.

        

            word: a word to search for

        """

        self.ans.clear()

        temp = self.trie

        text = ""

        for letter in word:

            if letter in temp:

                temp = temp[letter]

                text += letter

        remaining = list(temp)

        text_arr = [text] * len(remaining)

        for i in range(len(remaining)):

            if remaining[i] == '*':

                self.ans[text_arr[i]] = temp['*']

            else:

                self.find_similar_words(temp, remaining[i], text_arr[i])
# Reads the video into video variable

video = editor.VideoFileClip('/kaggle/input/video-data/Pixel.mp4')

# extract Audio video

converted_audio = video.audio



# by writing onto a wav file creating hormony in fps(44100)

converted_audio.write_audiofile('spotify.wav', logger = None, verbose = False)

audio = editor.AudioFileClip('spotify.wav')



# specch recognizer

r = sr.Recognizer()
chunk_duration = 10 # duration of each chunk





out_text = ""

obj = Trie()



# divides the while duration into equal length parts of size 10

spaces = linspace(0, audio.duration, int(ceil(audio.duration/chunk_duration)))

# utility variables for time conversion

hrs, min, sec = 0, 0, 0



# recognition process

if audio.duration > chunk_duration:

    for i in range(len(spaces)-1):

        audio.subclip(spaces[i], spaces[i+1]).write_audiofile('out.wav', verbose=False, logger=None)

        sec += chunk_duration

        if sec >= 60:

            min += 1

            sec -= 60

        if min >= 60:

            hrs += 1

            min -= 60

        with sr.AudioFile('out.wav') as source:

            audio_r = r.listen(source)

            try:

                sentence = r.recognize_google(audio_r)

                sentence = sentence.lower()

                obj.make_trie(sentence.split(), time="%02d:%02d:%02d"%(hrs, min, sec))

                out_text += "%02d:%02d:%02d"%(hrs, min, sec)+" "+sentence+"\n\n"

            except:

                pass

else:

    audio.write_audiofile('out.wav', verbose=False, logger = None)

    with sr.AudioFile('out.wav') as source:

        audio_r = r.listen(source)

        try:

            sentence = r.recognize_google(audio_r)

            sentence = sentence.lower()

            obj.make_trie(sentence.split(" "), time=seconds)

            out_text += "%02d:%02d:%02d"%(hrs, min, sec)+" "+sentence+"\n\n"

        except:

            pass

# Debug output

print("Text: ", out_text)
obj.find_words('pixel')

for i in obj.ans:

    print(i,':')

    print(sorted(obj.ans[i]))
# audio.subclip(0, 5).write_audiofile('out_1.wav')

# print(audio.duration)

# print(linspace(0, audio.duration, int(ceil(audio.duration/5))))

# audio.subclip(5, 400000).write_audiofile('out_2.wav')