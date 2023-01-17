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
file = open('/kaggle/input/sherlock.txt')
filestring = file.read()    #assign filestring as one giant string with the whole file

wordlist = filestring.split()  #split the string into a list
print (f"wordlist has {len(wordlist)} words in it.")
#print (wordlist)
# This is a function that will replace (most) punctuation
def stripWordPunctuation(word):
    return word.strip(".,()<>\"\\'~?!-;*/")


# For example item number 28 contains a period. We want to remove that.
print(wordlist[28])
print(stripWordPunctuation(wordlist[28]))

# We can also convert all of the uppercase character to lowercase, so that we can 
# count words consistently, whether or not they are capitalized.
print(wordlist[0])
print(wordlist[0].lower())
### Code to count the frequency of all words and print
### all of the words that appear more than 50 times
### and the number of times they appear

obviouscount = 0
elementarycount = 0

d = {}  # create a dictionary to keep track of the frequency that words occur -> {word : count}

# Process the file by going through every word in our wordlist
for word in wordlist:
    word = stripWordPunctuation(word.lower())
    
# The get() method returns the value of the item with the specified key. Syntax: dictionary.get(keyname, value).
# If there is no value, set the default value to: 0

    d[word] = d.get(word,0)+1 

#  In plain English the previous line says:
#  (and remember, we are in a FOR loop going through the entire wordlist, one word at a time)
#  For whatever word we are currently working on in ourdictionary, the value associated with that word
#  should assigned a value. So GET whatever is the current value of that word we are working on, and add ONE to it (+1)"
    
    
print(f"Elementary count: { d['elementary'] }")
print(f"Obvious count: { d['obvious'] }")   

for word in d:
    if d[word] > 50:
        print(f"{word}: {d[word]}")
        
#Code to print the word that appears most frequently
maxcount = 0
for word in d:
    if d[word] > maxcount:
        maxcount = d[word]
        maxword = word

print(f"\nThe word \"{maxword}\" appeared most often: {maxcount} times")
## Maximum of a list of numbers
max_so_far = 0
vals = d.values()
for v in vals:
    if v > max_so_far:
        max_so_far = v
print(max_so_far)
print(f"{maxword} appeared most often, {maxcount} times.")
