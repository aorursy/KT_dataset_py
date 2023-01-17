#Importing libraries 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time #to time running of certain cells 
#Importing Files and formatting 



df = pd.read_csv("../input/billboard-lyrics/billboard_lyrics_1964-2015.csv",encoding='latin-1')

import warnings

warnings.filterwarnings('ignore') #added in at the end, to ensure any warnings that did occur were not any issues 



#Converting lyrics series into string format 

df['Lyrics'] = df['Lyrics'].astype(str) 



#removing whitespaces from the lyrics column, at the start of each string and at the end 

df.Lyrics=df.Lyrics.str.strip() 



df.head()
# Counting how many songs in data which do not have lyrics. 

print('There are '+str(df[(df.Lyrics == "nan") |  (df.Lyrics =='') | (df.Lyrics == "NA")].Lyrics.count())+' songs without any lyrics. Below is the distribution of these by year.')



# Plotting the distribution of missing lyrics for each year 

from matplotlib.pyplot import figure



#Setting the size of the chart

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')



plt.xlabel('no of missing lyrics')

plt.ylabel('Year')



df[(df.Lyrics == "nan") |  (df.Lyrics =='') | (df.Lyrics == "NA")].Year.value_counts().plot.barh(title=

                                                    'Distribution of missing Lyrics')
#Splitting artist title into main and featuring artist 



#importing regular expressions library 

import re 



#Creating the 2 additional columns in our dataframe 





start_time = time.time()



#Below, the \sfeat\w* is the regular expresion for any instance of ' feat *any alpha-numeric character*'



df['Artist_main'] = df.Artist.apply(lambda x: re.split(r'\sfeat\w*',x,maxsplit=1)[0] if x != 'nan' else "")



df['Artist_featuring'] = df.Artist.apply(lambda x: re.split(r'\sfeat\w*',x,maxsplit=1)[1] if len(re.split(r'\sfeat\w*',x,maxsplit=1))==2 else "")



print("--- %s seconds ---" % (time.time() - start_time))



df.head()
#Creating a subset of data with no lyrics

no_lyrics=df[(df.Lyrics == "nan") |  (df.Lyrics =='') | (df.Lyrics == "NA")]



#Reset index, but save original index as series so we can refer back to this.

no_lyrics=no_lyrics.reset_index()   



no_lyrics.info()



no_lyrics.head()
!pip install lyricsgenius
#https://www.storybench.org/download-song-lyrics-genius-using-python/ Link with info on using API for lyrics genius 



import lyricsgenius #importing Genius API



# Genius API userid 

genius = lyricsgenius.Genius("lAHRp22UjrNpUD6BGztbbaeJqQtRZv8to5oVUsxDvhQOmktrglFgpUCvFixMhpbr")



############ 



# Each time one uses the genius API for search, it prints the result, to avoid this we will use the below custom function, 

# taken from: https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python



import os, sys



class HiddenPrints:

    def __enter__(self):

        self._original_stdout = sys.stdout

        sys.stdout = open(os.devnull, 'w')



    def __exit__(self, exc_type, exc_val, exc_tb):

        sys.stdout.close()

        sys.stdout = self._original_stdout



############



#Searching for lyrics



#In addition to lyrics, we will save the artist name and song name output from the search results to our dataframe.

# This is so we can check if the lyrics retrieved are for the right song and right artist 



import time

start_time = time.time()



r=range(250)



no_lyrics['Genius_Artist']=""

no_lyrics['Genius_Song']=""



for i in r:

    

    with HiddenPrints(): 

        song=genius.search_song(no_lyrics.Song[i],artist=no_lyrics.Artist_main[i])

    try:

        no_lyrics.Lyrics[i]=song.lyrics         #We will pull the resulting artist and song name as well as the lyrics.

        no_lyrics.Genius_Artist[i]=song.artist

        no_lyrics.Genius_Song[i]=song.title 

    except AttributeError:

        no_lyrics.Lyrics[i]=""

        no_lyrics.Genius_Artist[i]=""

        

print("--- %s seconds ---" % (time.time() - start_time))



print('The API search has managed to retrieve lyrics for '+str((no_lyrics['Lyrics'].values != '').sum())+ ' songs.')
no_lyrics.head(10)
#An example of how sensitive the genius search API is: 



genius.search_song(no_lyrics.Song[2],artist=no_lyrics.Artist_main[2])
from fuzzywuzzy import fuzz



#making all the strings lowercase, as this will reduce noise when computing string similarities 

no_lyrics=no_lyrics.apply(lambda x: x.astype(str).str.lower())



start_time = time.time()



#Similarity for the song names in our dataset compared with Genius song result 

no_lyrics['Song_Similarity']=no_lyrics.apply(lambda row: fuzz.token_set_ratio(row['Song'],row['Genius_Song'])/100,axis=1)



#Similarity for the artist names in our dataset compared with Genius artist result 

no_lyrics['Artist_Similarity']=no_lyrics.apply(lambda row: fuzz.token_set_ratio(row['Artist_main'],row['Genius_Artist'])/100,axis=1)



print("--- %s seconds ---" % (time.time() - start_time))



no_lyrics.head()
#PLotting similarity scores 



fig, axes = plt.subplots(1, 2)



axes[0].set_title('Artist similarity dist')

axes[1].set_title('Song similarity dist')

axes[0].set(xlabel="                         Similarity Score", ylabel="Frequency")



no_lyrics.Artist_Similarity.hist(bins=10,ax=axes[0])

no_lyrics.Song_Similarity.hist(bins=10,ax=axes[1])

#Splitting df for those results with similarity below 0.5 for both song and artist name - LOW SCORES 



no_lyrics[(no_lyrics.Artist_Similarity<0.5) & (no_lyrics.Song_Similarity<0.5) & (no_lyrics.Song_Similarity!=0)][['Song','Genius_Song','Artist','Genius_Artist','Artist_Similarity','Song_Similarity']]
#Splitting df for those results with similarity above 0.5 for both song and artist name - HIGH SCORES



no_lyrics[(no_lyrics.Artist_Similarity>0.5) & (no_lyrics.Song_Similarity>0.5) & (no_lyrics.Song_Similarity!=0)][['Song','Genius_Song','Artist','Genius_Artist','Artist_Similarity','Song_Similarity']]



# Note, there may be instances where the artist similarity is very high and the song similarity is very low, 

# or vice versa. For example, if there is a song that has been sung by multiple artists. 

# We are avoiding the instances of this by only taking lyrics where the 

# similarity score is high for BOTH artist and song name. 
#Creating a further subset, of data with the lyrics extracted. 



lyrics2clean=no_lyrics[(no_lyrics.Artist_Similarity>0.5) & (no_lyrics.Song_Similarity>0.5)] 



lyrics2clean=lyrics2clean.reset_index(drop=True)



lyrics2clean.info()



print('\nOut of the '+str(len(no_lyrics.index))+' songs we did not have lyrics for, we have managed to retrieve '+str(len(lyrics2clean.index))+".")
lyrics2clean.Lyrics[0]
#part of the code below taken from here: https://kvsingh.github.io/lyrics-sentiment-analysis.html

#we have improved the code by converting it to a function, this improves the efficiency of the code significantly. 



import re 



def clean_lyrics(lyric):

    

    #removing anything entailed in {} [] and (), as well as the parenthesis themselves, replacing them with "" 

    lyric= re.sub(r'[\(\[\{].*?[\)\]\}]',"",lyric)

    #removing new lines

    lyric= re.sub(r'\n'," ",lyric)

    #removing double white spaces 

    lyric= re.sub(r'\s{2}',"",lyric)

    lyric= lyric.replace("'","")

    lyric= lyric.replace(",","")

    lyric= lyric.lower()

    return lyric 



start_time = time.time()



lyrics2clean['Lyrics']=lyrics2clean['Lyrics'].apply(clean_lyrics)



print("--- %s seconds ---" % (time.time() - start_time))
lyrics2clean.Lyrics[0] #lyrics now cleaned up
#A reminder, in our lyrics2clean dataframe, we have the index for each song from the original df. 

#We left these in the subset so we can use these as references when filling in the missing lyrics in the original df. 



#converting the index series in the subset as integers 

lyrics2clean['index']=lyrics2clean['index'].astype(int)



#creating a dictionary with the key as original index number, and value as the lyrics for that index 

index_lyrics_dict = dict( zip( lyrics2clean['index'].values, lyrics2clean['Lyrics'].values) )





#Using numpy vector 'IF' approach to fill in the missing lyrics we now have 

#The first argument is the condition, i.e. if in the Lryics column there are missing lyrics, then

#set this to the value in the dictionary we've created IF the index for that particular song is in the dictionary,

#For those songs that have lyrics, no changes are made. 



df.iloc[:,4]  = np.where( ( (df.iloc[:,4] == "") | (df.iloc[:,4] == 'NA') | (df.iloc[:,4] == "nan") ), df.index.map(index_lyrics_dict) ,df.iloc[:,4])



df['Lyrics'] = df['Lyrics'].astype(str)
print('There are now '+str(df[(df.Lyrics == "nan") |  (df.Lyrics =='') | (df.Lyrics == "NA")].Lyrics.count())+' songs without any lyrics.')