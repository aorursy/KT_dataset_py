# The notebook helps to figure out certain songs from a string initial letters of

# their lyrics. For example, 'TOMTOMTMOTOMIBGIADOT' would be Take on me by a-ha.



# Oskar Niemenoja, Oct 2020



# Load the required libraries. For the data set we use the publicly available song-lyrics data set

# https://www.kaggle.com/terminate9298/songs-lyrics?select=lyrics.csv

# https://www.kaggle.com/edenbd/150k-lyrics-labeled-with-spotify-valence

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv



import string



import difflib
# Load the first data set



data = pd.read_csv('/kaggle/input/songs-lyrics/lyrics.csv', sep=',')
# Create a dict of the songs where each lyric has been truncated to its' initials



song_initials = {}



for index, row in data.iterrows():

    # Create a readable key

    artist_song = row.artist + ': ' + row.song_name

    

    # Create the lyric string of initial letters

    song_initials[artist_song] = "".join([word[0].upper() for word in row['lyrics'].split() if word[0].isalpha()])
songs_to_solve = [

    'DYWDYWCSHGAPFYDYWDYWNY',

    'CTITTNANOGSYFTBATS',

    'DTMINWTFYCTMINWDFYKITEIDIDIFY',

    'SPTFIAIBBSFILWAEM',

    'WGDDIAERASWGDS'

]



results = {}



for song in songs_to_solve:

    results[song] = [key for key in song_initials if song_initials[key].find(song) != -1]

results
# Load a larger data set



data_large = pd.read_csv('../input/150k-lyrics-labeled-with-spotify-valence/labeled_lyrics_cleaned.csv', sep=',')
# Same as above, some column names have changed places



song_initials_large = {}



for index, row in data_large.iterrows():

    # Create a readable key

    artist_song = row.artist + ': ' + row.song

    

    # Create the lyric string of initial letters

    song_initials_large[artist_song] = "".join([word[0].upper() for word in row['seq'].split() if word[0].isalpha()])
results_large = {}



for song in songs_to_solve:

    results_large[song] = [key for key in song_initials_large if song_initials_large[key].find(song) != -1]

results_large
# Taken from https://stackoverflow.com/questions/55271961/finding-the-closest-sub-string-by-hamming-distance



def ham_dist(s1, s2):

    if len(s1) != len(s2):

        raise ValueError("Undefined")

    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))



def search_min_dist(source,search):

    l = len(search)

    index = 0

    min_dist = l

    min_substring = source[:l]    

    for i in range(len(source)-l+1):

        d = ham_dist(search, source[i:i+l])

        if d<min_dist:

            min_dist = d

            index = i

            min_substring = source[i:i+l]  

    return (index,min_dist,min_substring)
# As above, but change the exact match to a hamming distance with distance

# fewer than two.



near_results = {}



for song in songs_to_solve:

    near_results[song] = [key for key in song_initials if search_min_dist(song_initials[key],song)[1] < 3]



near_results_large = {}



for song in songs_to_solve:

    near_results_large[song] = [key for key in song_initials_large if search_min_dist(song_initials_large[key],song)[1] < 2]

near_results
near_results_large