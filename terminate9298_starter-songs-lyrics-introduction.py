import seaborn as sns

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import regex as re

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

albums = pd.read_csv('/kaggle/input/album_details.csv')

display(albums.sample(10))

display(albums.shape)
songs = pd.read_csv('/kaggle/input/songs_details.csv')

display(songs.sample(10))

display(songs.shape)
def clean(song):    

    # Contractions

    song = song.lower()

    

    song = re.sub(r"he's", "he is", song)

    song = re.sub(r"there's", "there is", song)

    song = re.sub(r"we're", "we are", song)

    song = re.sub(r"that's", "that is", song)

    song = re.sub(r"won't", "will not", song)

    song = re.sub(r"they're", "they are", song)

    song = re.sub(r"can't", "cannot", song)

    song = re.sub(r"wasn't", "was not", song)

    song = re.sub(r"aren't", "are not", song)

    song = re.sub(r"isn't", "is not", song)

    song = re.sub(r"what's", "what is", song)

    song = re.sub(r"haven't", "have not", song)

    song = re.sub(r"hasn't", "has not", song)

    song = re.sub(r"there's", "there is", song)

    song = re.sub(r"he's", "he is", song)

    song = re.sub(r"it's", "it is", song)

    song = re.sub(r"you're", "you are", song)

    song = re.sub(r"i'm", "i am", song)

    song = re.sub(r"shouldn't", "should not", song)

    song = re.sub(r"wouldn't", "would not", song)

    song = re.sub(r"i'm", "i am", song)

    song = re.sub(r"isn't", "is not", song)

    song = re.sub(r"here's", "here is", song)

    song = re.sub(r"you've", "you have", song)

    song = re.sub(r"we're", "we are", song)

    song = re.sub(r"what's", "what is", song)

    song = re.sub(r"couldn't", "could not", song)

    song = re.sub(r"we've", "we have", song)

    song = re.sub(r"who's", "who is", song)

    song = re.sub(r"y'all", "you all", song)

    song = re.sub(r"would've", "would have", song)

    song = re.sub(r"it'll", "it will", song)

    song = re.sub(r"we'll", "we will", song)

    song = re.sub(r"we've", "we have", song)

    song = re.sub(r"he'll", "he will", song)

    song = re.sub(r"y'all", "you all", song)

    song = re.sub(r"weren't", "were not", song)

    song = re.sub(r"didn't", "did not", song)

    song = re.sub(r"they'll", "they will", song)

    song = re.sub(r"they'd", "they would", song)

    song = re.sub(r"don't", "do n't", song)

    song = re.sub(r"they've", "they have", song)

    song = re.sub(r"i'd", "i would", song)

    song = re.sub(r"You\x89Ûªre", "You are", song)

    song = re.sub(r"where's", "where is", song)

    song = re.sub(r"we'd", "we would", song)

    song = re.sub(r"i'll", "i will", song)

    song = re.sub(r"weren't", "were not", song)

    song = re.sub(r"they're", "they are", song)

    song = re.sub(r"let's", "let us", song)

    song = re.sub(r"it's", "it is", song)

    song = re.sub(r"can't", "cannot", song)

    song = re.sub(r"don't", "do not", song)

    song = re.sub(r"you're", "you are", song)

    song = re.sub(r"i've", "I have", song)

    song = re.sub(r"that's", "that is", song)

    song = re.sub(r"i'll", "i will", song)

    song = re.sub(r"doesn't", "does not", song)

    song = re.sub(r"i'd", "i would", song)

    song = re.sub(r"didn't", "did not", song)

    song = re.sub(r"ain't", "am not", song)

    song = re.sub(r"you'll", "you will", song)

    song = re.sub(r"i've", "i have", song)

    song = re.sub(r"don't", "do not", song)

    song = re.sub(r"i'll", "i will", song)

    song = re.sub(r"i'd", "i would", song)

    song = re.sub(r"let's", "let us", song)

    song = re.sub(r"you'd", "you would", song)

    song = re.sub(r"it's", "it is", song)

    song = re.sub(r"ain't", "am not", song)

    song = re.sub(r"haven't", "have not", song)

    song = re.sub(r"could've", "could have", song)

    song = re.sub(r"youve", "you have", song)

    song = re.sub(r"ev'ry" , 'every' , song)

    song = re.sub(r"coz" , 'because' , song)

    song = re.sub(r"n\'t" , 'not' , song)

    song = re.sub(r"that'll", "that will" ,song)

    song = re.sub(r"-", "" ,song)

    song = re.sub(r"[\.]+" , " . ", song)

    song = re.sub(r"\[[\sa-z0-9:-]*\]" , " ",song)

#     song = re.sub(r"\([a-zA-Z0-9\s\W]*\)" , " ", song)

    song = re.sub(r"\n\r\n", "\n\n", song)

    song = re.sub(r"\r\n", "\n\n", song)

    song = re.sub(r"\n\n\n\n", "\n\n", song)

    song = re.sub(r"\n\n\n", "\n\n", song)

    song = re.sub(r"\n\n", " <PARA> ", song)

    song = re.sub(r"\n", " <LINE> ", song)

    song = re.sub(r"â\x80\x98" , "'" , song)

    song = re.sub(r"â\x80\x99" , "'" , song)

    song = re.sub(r"â\x80¦" , "" , song)

    

    # Words with punctuations and special characters

    punctuations = ',@#!?+&*-%./();$=|{}^' + "`"

    remove_punctions = "'[]:?!()" + '"'

    for p in punctuations:

        song = song.replace(p, f' {p} ')

    for p in remove_punctions:

        song = song.replace(p , '')

    return song
class lyricsClass:

    def __init__(self , lyrics_filename = '/kaggle/input/lyrics.csv'):

        self.lyrics_filename = lyrics_filename

        self.read_lyrics()

        self.clear_text() # cleaning and Preparing the Lyrics

    

    # Reading the Lyrics File

    def read_lyrics(self):

        self.lyrics = pd.read_csv(self.lyrics_filename)

   

    # Display the Details

    def print_sample(self , size = 10):

        display(self.lyrics.sample(size))

    def print_info(self):

        display(self.lyrics.info())

    def print_random_lyrics(self , size = 1):

        print(self.lyrics.sample(size).values)

    

    #filtering the Text to New Column

    def clear_text(self):

        self.lyrics['clearLyrics'] = self.lyrics.lyrics.apply(lambda x: clean(x))

        

lyrics = lyricsClass()

sns.set_style("darkgrid")
lyrics.print_random_lyrics()
lyrics.lyrics
plt.figure(figsize=(15,35))

plt.yticks(rotation=10)

plt.xticks(rotation=10)

ax = sns.countplot(y = 'singer_name' , data = songs, order = songs['singer_name'].value_counts().index )

for p in ax.patches:

    height = p.get_width()

    ax.text(height+20,p.get_y()+p.get_height()/1.2,

            '{:1.2f}%'.format(height/songs.shape[0]*100),

            ha="center" , fontsize = 'small') 

    if height>30:

        ax.text(height-10,p.get_y()+p.get_height()/1.2,

            '{}'.format(height),

            ha="center", fontsize = 'small') 

plt.show()