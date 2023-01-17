# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

allSongs = pd.read_csv("../input/billboard_lyrics_1964-2015.csv", encoding = 'latin1')

allLyrics = allSongs['Lyrics'];

allLyricsAsWords = allLyrics.str.split(' ').str.join(' ')

print(allLyricsAsWords.size);

#print (len(allLyricsAsWords[1]))

j = 0;

totalStringLength = 0

totalString = ''

for i in range(1 , allLyricsAsWords.size):

    try :

        totalStringLength += len(allLyricsAsWords[i])                 

        totalString += allLyricsAsWords[i]

    except TypeError:

        a = 1

        #print('Not a string.. Ignoring')

        



#print(totalStringLength)

#print(list(totalString));

#allLyricsAsWords.to_csv('allLyricsAsWords.csv')



#out = open('output_1_test_lyrics.csv', "w")

#out.write(totalString)

        #for 

# Any results you write to the current directory are saved as output.


#print(check_output(["ls", "../working"]).decode("utf8"))

#print(type(totalString))

allLyricsAsLetters = list(totalString)



links = {}

linksArray = [];

for i in range(0, len(allLyricsAsLetters)-1):

    consecutiveLetters = allLyricsAsLetters[i]+allLyricsAsLetters[i+1]

    if (links.get(consecutiveLetters) == None):

        links[consecutiveLetters] = { 'count': 0 }

    linksArray.append(consecutiveLetters)

    links[consecutiveLetters]['count'] += 1

    #links.add(allLyricsAsLetters[i]+allLyricsAsLetters[i+1])

#print(pd.Series(allLyricsAsLetters))

#allLetter = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','']
    consecutiveLettersDataFrame = pd.DataFrame(linksArray);

consecutiveLettersOccurenceCount = consecutiveLettersDataFrame[0].value_counts()

print(consecutiveLettersOccurenceCount.head(10))
consecutiveLettersOccurenceCount.to_csv('consecutiveLettersCount_output.csv')
print(check_output(["ls", "../working"]).decode("utf8"))
allLyricsAsLettersDataFrame = pd.DataFrame(allLyricsAsLetters)

allLyricsAsLettersDataFrame[0].value_counts().to_csv('letterCounts.csv')