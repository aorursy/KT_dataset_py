import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import os
from urllib.request import urlopen
print(os.listdir("../input"))
fin = open('../input/words.txt')
fin.readline()
fin.readline()
line = fin.readline()
word = line.strip()
word
fin = open('../input/words.txt')
for line in fin:
    word = line.strip()
    print(word)
def uses_all(word, required):
    for letter in required: 
        if letter not in word:
            return False
    return True
uses_all("polly", "ol")
def uses_only(word, available):
    for letter in word: 
        if letter not in available:
            return False
    return True
uses_only("play", "alopy")
def is_palindrome(word):
    i = 0
    j = len(word)-1

    while i<j:
        if word[i] != word[j]:
            return False
        i = i+1
        j = j-1

    return True
is_palindrome("lol")