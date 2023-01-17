"""

Script:  JBR_NLP.py

Purpose:  Extends and customizes the analyze_book1.py code example from Think Python, 2nd Edition, by Allen Downey (http://thinkpython2.com)

http://thinkpython2.com/code/analyze_book1.py

"""



from __future__ import print_function, division



import random

import string

import re

import os

import nltk

from nltk.stem import WordNetLemmatizer





def process_file(filename, wordHist, skip_header):

    """Makes a wordHistogram that contains the words from a file.



    filename: string

    skip_header: boolean, whether to skip the Gutenberg header

   

    returns: map from each word to the number of times it appears.

    """

    try:

        fp = open(filename)

    except:

        print("\nDOUBLE CHECK OPENING: ", filename)



    if skip_header:

        skip_header(fp)



    try:    

        for line in fp:

            process_line(line, wordHist)

    except:

        print("\nDOUBLE CHECK PROCESSING LINES IN", filename)



    return wordHist





def skip_header(fp):

    """Reads from fp until it finds the line that ends the header



    fp: open file object

    """

    for line in fp:

        if line.startswith('*END*THE SMALL PRINT!'):

            break



def process_line(line, wordHist):

    """Adds the words in the line to the wordHistogram.



    Modifies wordHist.



    line: string

    wordHist: wordHistogram (map from word to frequency)

    """

    strippables = string.punctuation + string.whitespace + string.digits



    for word in line.split():

        # remove punctuation and convert to lowercase

        word = word.strip(strippables)

        word = re.sub("/", " ", word)

        word = word.lower()



        # remove URLs

        if word.startswith('http') or word.startswith('https') or word.startswith('www.'):

            continue

        elif word.endswith('.pdf') or word.endswith('online') or word.endswith('.org'):

            continue

        elif '.....' in word:

            continue

        elif '-' in word:

            continue

        else:

            # update the wordHistogram with root word

            wordHist[word] = wordHist.get(word, 0) + 1

 

def most_common(wordHist):

    """Makes a list of word-freq pairs in descending order of frequency.



    wordHist: map from word to frequency



    returns: list of (frequency, word) pairs

    """

    c = []

    for key, value in wordHist.items():

        c.append((value, key))



    c.sort()

    c.reverse()

    return c



def print_most_common(wordHist, num=10):

    """Prints the most commons words in a wordHistgram and their frequencies.

    

    wordHist: wordHistogram (map from word to frequency)

    num: number of words to print

    """

    c = most_common(wordHist)

    print("\n")

    print(num, ' MOST COMMON WORDS:')

    for freq, word in c[:num]:

        print(word, '\t', freq)



def subtractDict(d1, d2):

    """Returns a dictionary with all keys that appear in d1 but not d2



    d1, d2: dictionaries

    """

    res = {}

    for key, value in d1.items():

        if key not in d2:

            res[key] = value

    return res



def total_words(wordHist):

    """Returns the total of the frequencies in a wordHistogram."""

    return sum(wordHist.values())





def different_words(wordHist):

    """Returns the number of different words in a wordHistogram."""

    return len(wordHist)





def main():

    lemmatizer = WordNetLemmatizer()

    grammarHist = {}

    grammarWords = process_file('../input/jobbulletindata/JBR_Resources/grammarWords.txt', grammarHist, skip_header=False)

    JobBulletins = [os.path.join(root, file) for root, folder, JobBulletin in os.walk('../input/jobbulletindata/JobBulletins') for file in JobBulletin]



    # get stats for all Job Bulletins

    complexity = 10

    wordHist = {}

    for JobBulletin in JobBulletins:

        filename = re.sub('[\']','',JobBulletin)

        wordHist = process_file(filename, wordHist, skip_header=False)

    commonWords = most_common(wordHist)

    meaningWords = subtractDict(wordHist, grammarWords)

    longWords = [key for key,value in meaningWords.items() if len(key) > complexity]

    longWords.sort()

    fo = open("longwords.txt", 'w')

    print(longWords,file=fo)

    fo.close()



    fs = open("totWords.txt", 'w')

    print(total_words(wordHist),file=fs)

    print(different_words(wordHist),file=fs)

    fs.close()



    print("JOB BULLETINS", len(JobBulletins))

    print('\nTOTAL WORDS:', total_words(wordHist))

    print('\nTOTAL DIFFERENT WORDS:', different_words(wordHist))

    print_most_common(wordHist, 50)

    print('\nTOTAL COMPLEX WORDS:', len(longWords))

    #print('\nCOMPLEX WORDS:', longWords)

    

    # get stats for each Job Bulletin

    print("\nComputing statistics for each Job Bulletin")

    wordHist = {}

    fileStatsRow = {}

    fileStats = []

    totWords = 0

    difWords = 0

    totLongWords = 0

    for JobBulletin in JobBulletins:



        wordHist = {}

        filename = re.sub('[\']','',JobBulletin)                            # strip the single quote so the file will open

        wordHist = process_file(filename, wordHist, skip_header=False)

        commonWords = most_common(wordHist)

        longWords = subtractDict(wordHist, grammarWords)

        longWords = [key for key,value in longWords.items() if len(key) > complexity]

        longWords.sort()



        totWords += total_words(wordHist)

        difWords += different_words(wordHist)

        totLongWords +=  len(longWords)



        fileStatsRow['FILE_NAME'] = filename

        fileStatsRow['TOT_WORDS']= total_words(wordHist)

        fileStatsRow['TOT_DIF_WORDS'] = different_words(wordHist)

        fileStatsRow['TOT_LONG_WORDS'] = len(longWords)

        fileStatsRow['LONG_WORDS'] = longWords

        fileStats.append(fileStatsRow)

        fileStatsRow = {}



    import pandas as pd

    import numpy as np

    import matplotlib.pyplot as plt

    

    df = pd.DataFrame(fileStats)

    df.index.name = 'IDX'

    df.to_csv("fileStats.csv")



    print("\nStatistics for each file have been saved in fileStats.csv")



if __name__ == '__main__':

    main()